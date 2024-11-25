import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import numpy as np
import ipdb
from cmehr.models.mimic4.base_model import MIMIC4LightningModule
from torchdiffeq import odeint as odeint


def create_net(n_inputs, n_outputs, n_layers=0, n_units=10, nonlinear=nn.Tanh, add_softmax=False, dropout=0.0):
    if n_layers >= 0:
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.Dropout(p=dropout))

        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))
        if add_softmax:
            layers.append(nn.Softmax(dim=-1))

    else:
        layers = [nn.Linear(n_inputs, n_outputs)]

        if add_softmax:
            layers.append(nn.Softmax(dim=-1))

    return nn.Sequential(*layers)


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert (start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                             torch.linspace(start[i], end[i], n_points)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


class GRU_unit_cluster(nn.Module):
    def __init__(self, latent_dim, input_dim,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=100,
                 use_mask=False,
                 dropout=0.0):
        super(GRU_unit_cluster, self).__init__()

        if update_gate is None:
            if use_mask:
                self.update_gate = nn.Sequential(
                    nn.Linear(latent_dim + 2 * input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid())
            else:
                self.update_gate = nn.Sequential(
                    nn.Linear(latent_dim + input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid())
        else:
            self.update_gate = update_gate

        if reset_gate is None:
            if use_mask:
                self.reset_gate = nn.Sequential(
                    nn.Linear(latent_dim + 2 * input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid())
            else:
                self.reset_gate = nn.Sequential(
                    nn.Linear(latent_dim + input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid())
        else:
            self.reset_gate = reset_gate

        if new_state_net is None:
            if use_mask:
                self.new_state_net = nn.Sequential(
                    nn.Linear(latent_dim + 2 * input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                )
            else:
                self.new_state_net = nn.Sequential(
                    nn.Linear(latent_dim + input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                )
        else:
            self.new_state_net = new_state_net

    def forward(self, y_i, x):
        y_concat = torch.cat([y_i, x], -1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)

        concat = y_i * reset_gate

        concat = torch.cat([concat, x], -1)

        new_probs = self.new_state_net(concat)

        new_y_probs = (1 - update_gate) * new_probs + update_gate * y_i

        assert (not torch.isnan(new_y_probs).any())

        return new_y_probs


class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        self.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def init_network_weights(self, net, std=0.1):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t_local, y, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)


class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents,
                 odeint_rtol=1e-4, odeint_atol=1e-5, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
        # Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 2, 0, 3)

        assert (torch.mean(pred_y[:, :, 0, :] - first_point) < 0.001)
        assert (pred_y.size()[0] == n_traj_samples)
        assert (pred_y.size()[1] == n_traj)

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict,
                               n_traj_samples=1):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y


class DGM2OModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 latent_dim: int = 10,
                 input_dim: int = 30,
                 cluster_num: int = 20,
                 z0_dim=10,
                 n_gru_units=10,
                 use_sparse=False,
                 dropout=0.0,
                 use_mask=False,
                 use_static=True,
                 num_time_steps_and_static=[100, 2],
                 * args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)
        self.save_hyperparameters()

        rec_ode_func = ODEFunc(
            input_dim=10,
            latent_dim=10,
            ode_func_net=create_net(10, 10))
        z0_diffeq_solver = DiffeqSolver(
            10, rec_ode_func, "euler", 10, odeint_rtol=1e-3, odeint_atol=1e-4)
        GRU_update = GRU_unit_cluster(
            10, input_dim, n_units=10, use_mask=False, dropout=0.0)

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        self.dropout = dropout

        if GRU_update is None:
            self.GRU_update = GRU_unit_cluster(latent_dim, input_dim,
                                               n_units=n_gru_units, use_mask=use_mask, dropout=dropout)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.cluster_num = cluster_num
        self.use_sparse = use_sparse
        self.use_mask = use_mask

        self.min_steps = 0.0

        self.extra_info = None

        self.concat_data = True

        if self.concat_data:
            self.infer_emitter_z = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
                nn.Linear(latent_dim + cluster_num, self.cluster_num),
                nn.Dropout(p=self.dropout)
            )
        else:
            self.infer_emitter_z = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
                nn.Linear(latent_dim, self.cluster_num),
                nn.Dropout(p=self.dropout)
            )

        self.infer_transfer_z = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.cluster_num, latent_dim),
            nn.Dropout(p=self.dropout)
        )

        self.decayed_layer = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(1, 1),
            nn.Dropout(p=self.dropout)
        )

        ts, static = num_time_steps_and_static
        # if use_static:
        #     self.mlp = nn.Linear(ts * 10 + static, 2)
        # else:
        #     self.mlp = nn.Linear(ts * 10 + static, 8)
        self.pred_layer = nn.Linear(10 + static, self.num_labels)

        import json
        with open("/home/*/Documents/MMMSPG/src/cmehr/preprocess/mimic3/mimic3models/resources/discretizer_config.json", "r") as f:
            config = json.load(f)
        variables = config["id_to_channel"]
        static_variables = ["Height", "Weight"]
        inp_variables = list(set(variables) - set(static_variables))
        self.static_indices = [variables.index(v) for v in static_variables]
        self.inp_indices = [variables.index(v) for v in inp_variables]

    def forward(self, x_ts, x_ts_mask, ts_tt_list,
                cxr_imgs_sequences=None,
                cxr_time_sequences=None,
                cxr_time_mask_sequence=None,
                labels=None, reg_ts=None):
        """ Forward function of Multimodal model
        :param x_ts: (B, N, D_t), torch.Tensor, time series data
        :param x_ts_mask: (B, N, D_t), torch.Tensor, time series mask
        :param ts_tt_list: (B, N), torch.Tensor, time series time
        :param labels: (B, ), torch.Tensor, labels
        :param reg_ts: (B, N_r, D_r), torch.Tensor, regular time series data
        """
        # time_steps = ts_tt_list
        time_steps = torch.arange(x_ts.shape[1]).type_as(x_ts).long()
        src = x_ts[:, :, self.inp_indices]
        static = x_ts[:, :, self.static_indices]
        maxlen, batch_size = src.shape[1], src.shape[0]
        src_mask = x_ts_mask[:, :, self.inp_indices]

        data = torch.cat([src, src_mask], dim=2)
        static_data = torch.mean(static, dim=1)

        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:, 0, :].unsqueeze(0)
            all_y_i = self.GRU_update(prev_y, xi)

            all_y_i = F.softmax(all_y_i.unsqueeze(0), -1)

            extra_info = None
        else:
            _, latent_y_states, extra_info = self.run_odernn(
                data, time_steps, run_backwards=False,
                save_info=False)

        # if save_info:
        #     self.extra_info = extra_info

        vec = latent_y_states.squeeze()
        vec = torch.permute(vec, (1, 0, 2))
        vec = torch.reshape(
            vec, (vec.size()[0], vec.size()[1], vec.size()[2]))

        vec = vec.mean(dim=1)
        if static_data is not None:     # add static data
            vec = torch.cat((vec, static_data), dim=1)

        logits = self.pred_layer(vec)
        if self.task in ['ihm', 'readm']:
            if labels != None:
                ce_loss = self.loss_fct1(logits, labels)
                return ce_loss
            return F.softmax(logits, dim=-1)[:, 1]

        elif self.task == 'pheno':
            if labels != None:
                labels = labels.float()
                ce_loss = self.loss_fct1(logits, labels)
                return ce_loss
            return torch.sigmoid(logits)

    def update_joint_probs(self, n_traj, joint_probs, t, latent_y_states, delta_t, full_curr_rnn_input=None):

        # 		n_traj, n_tp, n_dims = data.size()
        if full_curr_rnn_input is None:
            full_curr_rnn_input = torch.zeros((self.cluster_num, n_traj, self.cluster_num), dtype=torch.float,
                                              device=self.device)

            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros(
                    (n_traj, self.cluster_num), dtype=torch.float, device=self.device)
                curr_rnn_input[:, k] = 1
                full_curr_rnn_input[k] = curr_rnn_input

        z_t_category_infer_full = self.emit_probs(
            latent_y_states[t], full_curr_rnn_input, delta_t, t)

        updated_joint_probs = torch.sum(
            z_t_category_infer_full * torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)

        joint_probs_sum = torch.sum(updated_joint_probs)

        return updated_joint_probs

    def emit_probs(self, prev_y_state, prev_y_prob, delta_t, i):

        delta_t = delta_t.to(self.device)

        if len(prev_y_prob.shape) > 2:
            prev_y_state = prev_y_state.repeat(prev_y_prob.shape[0], 1, 1)

        if i > 0:
            delta_t = delta_t.float()
            decayed_weight = torch.exp(-torch.abs(
                self.decayed_layer(delta_t.view(1, 1))))

            decayed_weight = decayed_weight.view(-1)
        else:
            decayed_weight = 0.5

        if self.concat_data:
            input_z_w = torch.cat([prev_y_prob, prev_y_state], -1)
            prev_y_prob = F.softmax(self.infer_emitter_z(input_z_w), -1)
        else:
            prev_y_prob = F.softmax(self.infer_emitter_z(
                (decayed_weight * self.infer_transfer_z(prev_y_prob) + (1 - decayed_weight) * prev_y_state)), -1)

        return prev_y_prob

    def run_odernn_single_step(self, data, time_steps, full_curr_rnn_input=None,
                               run_backwards=False, save_info=False, prev_y_state=None):

        extra_info = []

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        n_traj = data.size()[1]
        if prev_y_state is None:
            prev_y_state = torch.zeros(
                (1, n_traj, self.latent_dim)).to(self.device)

        prev_t, t_i = time_steps[0], time_steps[1]

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        if (t_i - prev_t) < self.min_steps:
            time_points = torch.stack((prev_t, t_i))
            inc = self.z0_diffeq_solver.ode_func(
                prev_t, prev_y_state) * (t_i - prev_t)

            assert (not torch.isnan(inc).any())

            ode_sol = prev_y_state + inc
            ode_sol = torch.stack((prev_y_state, ode_sol), 2).to(self.device)

            assert (not torch.isnan(ode_sol).any())
        else:
            n_intermediate_tp = max(2, ((t_i - prev_t) / self.min_steps).int())

            time_points = linspace_vector(prev_t, t_i, n_intermediate_tp)
            ode_sol = self.z0_diffeq_solver(prev_y_state, time_points)

            assert (not torch.isnan(ode_sol).any())

        if torch.mean(ode_sol[:, :, 0, :] - prev_y_state) >= 0.001:
            print("Error: first point of the ODE is not equal to initial value")
            print(torch.mean(ode_sol[:, :, 0, :] - prev_y_state))
            exit()

        yi_ode = ode_sol[:, :, -1, :]

        prev_y_state = self.GRU_update(yi_ode, data)

        xi = data[:, :].unsqueeze(0)

        if save_info:
            d = {"yi_ode": yi_ode.detach(),  # "yi_from_data": yi_from_data,
                 #  					 "yi": yi.detach(),
                 #  					 "yi_std": yi_std.detach(),
                 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
            extra_info.append(d)

        return prev_y_state

    def run_odernn(self, data, time_steps, run_backwards=False, save_info=False, exp_y_states=None):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        n_traj, n_tp, n_dims = data.size()

        extra_info = []

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        prev_y_prob = torch.zeros(
            (1, n_traj, self.cluster_num)).to(self.device)

        prev_y_state = torch.zeros(
            (1, n_traj, self.latent_dim)).to(self.device)

        joint_probs = torch.zeros(
            [n_tp, n_traj, self.cluster_num], dtype=torch.float, device=self.device)

        if not run_backwards:
            prev_t, t_i = time_steps[0] - 0.01, time_steps[0]
        else:
            prev_t, t_i = time_steps[-1], time_steps[-1] + 0.01

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = (time_steps[-1] - time_steps[0]
                        ) / (len(time_steps) * 0.5)

        self.min_steps = minimum_step

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        latent_ys = []

        latent_y_states = []

        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        first_ys = 0

        first_y_state = 0

        count = 0

        for i in time_points_iter:
            if (t_i - prev_t) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                inc = self.z0_diffeq_solver.ode_func(
                    prev_t, prev_y_state) * (t_i - prev_t)

                assert (not torch.isnan(inc).any())

                ode_sol = prev_y_state + inc
                ode_sol = torch.stack(
                    (prev_y_state, ode_sol), 2).to(self.device)

                assert (not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(
                    2, ((t_i - prev_t) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y_state, time_points)

                assert (not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :] - prev_y_state) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_y_state))
                exit()

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:, i, :].unsqueeze(0)
            prev_y_state = self.GRU_update(yi_ode, xi)

            if exp_y_states is not None:
                print(torch.norm(exp_y_states[:, count] - prev_y_state))

            latent_y_states.append(prev_y_state.clone())

            if not run_backwards:
                prev_t, t_i = time_steps[i], time_steps[(
                    i + 1) % time_steps.shape[0]]
            else:
                prev_t, t_i = time_steps[(i - 1)], time_steps[i]

            if save_info:
                d = {"yi_ode": yi_ode.detach(),  # "yi_from_data": yi_from_data,
                     #  					 "yi": yi.detach(),
                     #  					 "yi_std": yi_std.detach(),
                     "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)

            count += 1

        latent_y_states = torch.stack(latent_y_states, 1)

        prev_t, t_i = time_steps[0] - 0.01, time_steps[0]

        if run_backwards:
            latent_y_states = torch.flip(latent_y_states, [1])
        prev_y_prob = torch.zeros(
            (1, n_traj, self.cluster_num)).to(self.device)
        for t in range(latent_y_states.shape[1]):
            prev_y_state = latent_y_states[:, t]

            curr_prob = self.emit_probs(
                prev_y_state, prev_y_prob, t_i - prev_t, t)

            prev_y_prob = curr_prob

            latent_ys.append(prev_y_prob.clone())

            prev_t, t_i = time_steps[t], time_steps[(
                t + 1) % time_steps.shape[0]]

        latent_ys = torch.stack(latent_ys, 1)

        return latent_ys, latent_y_states, extra_info


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        file_path=str(ROOT_PATH / "output_mimic4/ihm"),
        tt_max=48,
        batch_size=4
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
        print(f"{k}: ", v.shape)

    """
    ts: torch.Size([4, 157, 17])
    ts_mask:  torch.Size([4, 157, 17])
    ts_tt:  torch.Size([4, 157])
    reg_ts:  torch.Size([4, 48, 34])
    input_ids:  torch.Size([4, 5, 128])
    attention_mask:  torch.Size([4, 5, 128])
    note_time:  torch.Size([4, 5])
    note_time_mask: torch.Size([4, 5])
    label: torch.Size([4])
    """
    model = DGM2OModule(
    )
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        labels=batch["label"],
    )
    print(loss)
