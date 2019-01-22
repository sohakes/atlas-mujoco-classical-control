import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils, spaces
from collections import deque
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
import glfw
import time
from scipy.interpolate import CubicSpline


class AtlasEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):        
        mujoco_env.MujocoEnv.__init__(self, './assets/atlas_v5.xml', 1, seed=0)

        #changes action space
        bounds = self.model.actuator_ctrlrange.copy()
        self.lin_vel = [0, 0, 0]

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None


    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def mass_center(self):
        mass = np.expand_dims(self.model.body_mass, 1)
        xipos = self.sim.data.xipos
        return (np.sum(mass * xipos, 0) / np.sum(mass))

    #Same reward as the one for humanoid, but I probably won't use it
    def step(self, a):
        action = a
        pos_before = self.mass_center()
        self.do_simulation(a, self.frame_skip)
        pos_after = self.mass_center()
        alive_bonus = 5.0
        data = self.sim.data
        self.lin_vel = (pos_after - pos_before) / self.model.opt.timestep
        lin_vel_cost = 0.25 * (pos_after[0] - pos_before[0]) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        quad_ctrl_cost = 0.1 * np.square(action).sum()
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus        
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 0.5) or (qpos[2] > 1.3))   
        done = False

        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        self.t = 0
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


def main():
    env = AtlasEnv()

    def find_route(to):
        idxlist = [env.model.body_jntadr[to]]
        if env.model.body_jntadr[to] == -1 or env.sim.model.joint_id2name(env.model.body_jntadr[to]) == 'main':
            return None
        last = to
        while env.model.body_parentid[last] != 0:
            theid = env.model.body_parentid[last]
            last = theid
            joint_idx = env.model.body_jntadr[theid]
            if joint_idx == -1:
                continue
            try: #such a hack, but I didn't find a method for joint->ctrl[]
                joint_name = env.sim.model.joint_id2name(joint_idx)
                ctrl_id = env.sim.model.actuator_name2id(joint_name + '_actuator')
            except ValueError:
                continue
            idxlist.insert(0, joint_idx)
            
            
        return idxlist

    #Page 58 from Shuuji Kajita's Introduction to Humanoid Robotics
    def calc_jacobian(idx_route, target_id):
        #In this case, I do want xipos since it's the center of mass
        target = env.sim.data.xipos[target_id]
        jsize = len(idx_route)
        J = np.zeros((6, jsize))
        mult = 1
        start = 0
        end = jsize
        for i in range(start, end): #Since I skip the root, it starts from 0 here (instead of from 1 like the book)
            j = idx_route[i]
            # I can get the joint axis directly with xaxis, need to get joint position before that
            joint_idx = j
            assert joint_idx != -1
            
            a = mult * env.sim.data.xaxis[joint_idx]

            #Here it's probably still xpos since it's the joint actuation center
            bodyid = env.model.jnt_bodyid[j]
            parentid = env.model.body_parentid[bodyid]
            
            v = np.cross( a, target - env.sim.data.body_xpos[bodyid])
            J[:, i] = np.concatenate([v, a])
        
        return J
    
    # https://www.cs.ubc.ca/~van/papers/2010-TOG-gbwc/paper.pdf
    def compensate_gravity(leg):
        assert leg == 'left' or leg == 'right' or leg == 'both'
        #get everything except the root
        torques = np.zeros(len(env.action_space.low))
        for i in range(1, len(env.sim.data.body_xpos)):
            idx_route = find_route(i)
            if idx_route is None:
                continue
            if (leg == 'left' and 'l_leg' in env.sim.model.joint_id2name(idx_route[-1])) or (leg == 'right' and 'r_leg' in env.sim.model.joint_id2name(idx_route[-1])):
                continue

            Ji = calc_jacobian(idx_route, i)[:3,:]
            Fi = -env.model.body_mass[i] * env.sim.model.opt.gravity
            torque = np.dot(Ji.T, Fi)
            
            for j in range(len(idx_route)):
                joint_idx = idx_route[j]
                joint_name = env.sim.model.joint_id2name(joint_idx)
                ctrl_id = env.sim.model.actuator_name2id(joint_name + '_actuator')
                torques[ctrl_id] += torque[j]
        return torques

    def position_pid_gen(kp=50, kd=None):
        kd = 2*np.sqrt(kp) if kd is None else kd
        last_joint_errors = None

        def position_pid(joint_spss):
            nonlocal kd
            nonlocal kp
            nonlocal last_joint_errors
            joint_pvs = []
            joint_sps = [(p % (2*np.pi)) for p in joint_spss]
            for c in range(len(env.sim.data.ctrl)):
                actuator_name = env.sim.model.actuator_id2name(c)
                joint_idx = env.sim.model.joint_name2id(actuator_name[:-9])
                qpos_joint_idx = env.sim.model.jnt_qposadr[joint_idx]
                joint_pvs.append(env.sim.data.qpos[qpos_joint_idx] % (2*np.pi))
            
            torques = []
            joint_errors = []
            for i in range(len(joint_pvs)):
                if joint_sps[i] is None: #no position control for this one
                    torques.append(0)
                    joint_errors.append(0)
                    continue
                joint_error = joint_sps[i] - joint_pvs[i]
                torque = kp * joint_error
                joint_errors.append(joint_error)
                if last_joint_errors is not None:
                    torque += kd * (joint_errors[i] - last_joint_errors[i]) /env.dt
                torques.append(torque)
            last_joint_errors = joint_errors

            return np.array(torques)
        return position_pid

    def get_step(v, h, g, alpha, vd):
        xd = v[0] * np.sqrt(h/g + v[0]**2/(4*(g**2)))
        yd = v[1] * np.sqrt(h/g + v[1]**2/(4*(g**2)))
        xdd = xd - alpha * vd
        return (xdd, yd)
        
    def get_step_env(env, alpha, vd):
        return get_step(env.lin_vel, env.mass_center()[2], -env.sim.model.opt.gravity[2], alpha, vd)

    def get_swing_foot_pos_gen(foot_height):
        cs = CubicSpline([0, 0.5, 1], [0.05, foot_height, 0.05])

        #if it behaves in a weird way, I should stop considering x0, y0 and use the current one
        def get_swing_foot_pos(phi, env, alpha, vd, x0, y0):
            nonlocal cs
            xd, yd = get_step_env(env, alpha, vd)
            x = (1 - phi) * x0 + phi * xd
            y = (1 - phi) * y0 + phi * yd
            z = cs(phi)
            return [x, y, z]

        return get_swing_foot_pos


    def get_contact(body_name, env):
        body_id = env.sim.model.body_name2id(body_name)
        for i in range(env.sim.data.ncon):
            contact = env.sim.data.contact[i]
            geom1_id= env.sim.data.contact[i].geom1
            geom2_id = env.sim.data.contact[i].geom2
            ground_id = env.sim.model.geom_name2id('ground')
            if geom1_id == ground_id:
                if env.sim.model.geom_bodyid[geom2_id] == body_id:
                    return True
            if geom2_id == ground_id:
                if env.sim.model.geom_bodyid[geom1_id] == body_id:
                    return True

        return False

    ### ALL IK CODE

    def rot2omega(R):
        el = np.array([ R[2,1] - R[1,2],
                        R[0,2] - R[2,0],
                        R[1,0] - R[0,1]])
        norm_el = np.linalg.norm(el)
        if norm_el > np.finfo(float).eps:
            return np.arctan2(norm_el, np.trace(R)-1)/norm_el * el
        elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
            return [0,0,0]
        else:
            return np.pi/2 * np.array([R[0,0]+1, R[1,1]+1,R[2,2]+1])


    def calc_V_WErr(rot, pos, Cnowid):
        perr = pos - env.sim.data.body_xpos[Cnowid]
        xmat = np.reshape(env.sim.data.body_xmat, (-1, 3, 3))
        Rerr = np.dot(xmat[Cnowid].T, rot)
        werr = np.dot(xmat[Cnowid], rot2omega(Rerr))
        return np.concatenate((perr, werr), axis=0)
    
    def fill_position_control(route=None):
        dqdif = np.zeros(len(env.sim.data.ctrl))
        #Only the size is necessary
        for joint_idx in range(len(env.sim.model.jnt_qposadr)):
            
            joint_name = env.sim.model.joint_id2name(joint_idx)
            ctrl_id = 0
            try:
                ctrl_id = env.sim.model.actuator_name2id(joint_name + '_actuator')
            except ValueError:
                continue
            qpos_joint_idx = env.sim.model.jnt_qposadr[joint_idx] #get the corresponding joint angle
            dqdif[ctrl_id] = env.sim.data.qpos[qpos_joint_idx] #assign the current joint pos
            if route is not None and joint_idx not in route:
                dqdif[ctrl_id] = None
        return dqdif

    def sum_dqdif(dqdif, dq, idx_route):
        for j, dqt in zip(idx_route, dq):
            joint_idx = j
            dqdif[joint_idx] += dqt
        return dqdif

    def move_joints(idx_route, dq):
        for j, dqt in zip(idx_route, dq):
            joint_idx = j
            qpos_joint_idx = env.sim.model.jnt_qposadr[joint_idx]
            env.sim.data.qpos[qpos_joint_idx] += dqt
            env.sim.data.qvel[qpos_joint_idx - 1] = 0
        env.sim.forward()

    def get_weight_joint_range(idx_route):
        ranges = [env.model.jnt_range[j] for j in idx_route]
        q_vector = [env.sim.data.qpos[env.sim.model.jnt_qposadr[j]] for j in idx_route]
        def normalize_range(mini, maxi, val):
            return ((val - mini) / (maxi - mini))

        norm_q_vector = [normalize_range(r[0], r[1], q) for r, q in zip(ranges, q_vector)]

        def weight_task(q):
            buffer = 0.05
            W0 = 0.8
            if q <= 0:
                return -W0
            if q >= 1:
                return W0
            if q < buffer:
                return -(W0 / 2) * (1 + np.cos(np.pi * (q / buffer)))
            if q > (1 - buffer):
                return (W0 / 2) * (1 + np.cos(np.pi * ((1 - q) / buffer)))
            return 0

        costs = [weight_task(q) for q in norm_q_vector]
        
        return np.diag(costs)

    def inverse_kinematics_LM(to, target_rot, target_pos, idx_route = [], from_node=None, reset_pos=False):
        qpos_o = np.copy(env.sim.data.qpos)
        qvel_o = np.copy(env.sim.data.qvel)
        if len(idx_route) == 0:
            if from_node is not None:
                idx_route = find_route_from_to(from_node, to)[1:]                
            else:
                idx_route = find_route(to)[:]

        dqdif_start = fill_position_control(idx_route)

        wn_pos = 1/0.3
        wn_ang = 1/(2*np.pi)
        We = np.diag([wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang])
        Wn = np.eye(len(idx_route))
        Jc = np.eye(len(idx_route))

        err = calc_V_WErr(target_rot, target_pos, to)
        Ek = np.dot(np.dot(err.T, We), err)
        
        for i in range(10):
            J = calc_jacobian(idx_route, to)
            lambda_v = Ek + 0.002

            Wc = get_weight_joint_range(idx_route)

            #Jh2 uses joint limits
            # Jh2 = np.dot(np.dot(J.T, We), J) + np.dot(np.dot(Jc.T, Wc), Jc) + np.dot(Wn, lambda_v)
            Jh = np.dot(np.dot(J.T, We), J) + np.dot(Wn, lambda_v)

            gerr = np.dot(np.dot(J.T, We), err)
            
            dq = np.linalg.lstsq(Jh, gerr)[0]

            move_joints(idx_route, dq)

            err = calc_V_WErr(target_rot, target_pos, to) 
            Ek2 = np.dot(np.dot(err.T, We), err)

            if Ek2 < 10**(-12):
                break
            elif Ek2 < Ek:
                Ek = Ek2
            else:
                move_joints(idx_route, -dq)
                break
        dqdif = fill_position_control(idx_route)
        if reset_pos:
            env.sim.data.qpos[:] = qpos_o
            env.sim.data.qvel[:] = qvel_o
        env.sim.forward()
        return dqdif, dqdif_start, idx_route

    def chain_ik(ik_funs):
        qvel_o = np.copy(env.sim.data.qvel)
        qpos_o = np.copy(env.sim.data.qpos)
        dqdif_start = fill_position_control()
        ret1 = None
        idx_route = []
        for ik_fun in ik_funs:
            ret1, ret2, r = ik_fun()
            idx_route += r
        
        ret = fill_position_control(idx_route)
        env.sim.data.qpos[:] = qpos_o
        env.sim.data.qvel[:] = qvel_o
        env.sim.forward()

        return ret, dqdif_start

    ## END IK CODE

    pid_fun = position_pid_gen(2000, 20)
    for i_episode in range(1):
        observation = env.reset()
        total_reward = 0

        stance_leg = 'left'
        left_foot_id = env.sim.model.body_name2id('l_foot')
        right_foot_id = env.sim.model.body_name2id('r_foot')
        stance_left_ik_route = [    env.sim.model.joint_name2id('l_leg_hpz'),
                                    env.sim.model.joint_name2id('l_leg_akx')]
        stance_right_ik_route = [   env.sim.model.joint_name2id('r_leg_hpz'),
                                    env.sim.model.joint_name2id('r_leg_akx')]

        

        foot_pos_gen = get_swing_foot_pos_gen(0.15)
        
        number_of_steps = 10
        time_each_step = 0.5

        t = 0
        xmat = np.copy(np.reshape(env.sim.data.body_xmat, (-1, 3, 3)))

        for step_number in range(number_of_steps):
            total_timesteps_in_step = int(time_each_step/env.dt)
            swing_leg_id = right_foot_id if stance_leg == 'left' else left_foot_id
            stance_leg_id = right_foot_id if stance_leg == 'right' else left_foot_id
            stance_leg_route = stance_right_ik_route if stance_leg == 'right' else stance_left_ik_route
            swing_foot_position_start = np.copy(env.sim.data.body_xpos[swing_leg_id])
            swing_foot_position_start[2] = 0.05
            stance_foot_position = np.copy(env.sim.data.body_xpos[stance_leg_id])
            stance_foot_position[2] = 0.05
            swing_foot_rot = np.copy(xmat[swing_leg_id])
            stance_foot_rot = np.copy(xmat[stance_leg_id])          
            
            for timestep_in_step in range(total_timesteps_in_step):
                foot_strike = get_contact('r_foot' if stance_leg == 'left' else 'l_foot', env)
                phi = timestep_in_step/total_timesteps_in_step
                if phi > 0.3 and foot_strike: #checks phi since there is some time for swinging the leg
                    break

                target_swing_foot_pos = foot_pos_gen(   phi, env, 0.05, 0.01, swing_foot_position_start[0],
                                                        swing_foot_position_start[1])

                #Use inverse kinematics for the stance foot to stay at the current position
                #it should probably use only the ankle and hip as the route, but maintain the position
                #in everything else (that could be a problem, since in the change swing->stance
                # it could happen that the leg will jerk to get to the correct position)
                p1, s, s1 = inverse_kinematics_LM(stance_leg_id,stance_foot_rot,
                                             stance_foot_position, reset_pos=True)
                p2, s, s2 = inverse_kinematics_LM(swing_leg_id,swing_foot_rot,
                                             target_swing_foot_pos, reset_pos=True)
                
                positions = []
                for a, b in zip(p1, p2):
                    assert np.isnan(a) or np.isnan(b)
                    if np.isnan(a) and np.isnan(b):
                        positions.append(0)
                    else:
                        positions.append(a if np.isnan(b) else b)

                env._get_viewer().add_marker(pos=stance_foot_position, size=[0.05, 0.05, 0.05], rgba=[1,0,0,1], label="stance", mat=stance_foot_rot)
                env._get_viewer().add_marker(pos=target_swing_foot_pos, size=[0.05, 0.05, 0.05], rgba=[0,1,0,1], label="swing", mat=swing_foot_rot)

                # don't compensate the stance leg
                position_torque = np.array(pid_fun(positions))

                action = position_torque + compensate_gravity(stance_leg)

                observation, reward, done, _ = env.step(positions)
                total_reward += reward
                env.render()

            stance_leg = 'left' if stance_leg == 'right' else 'right'

        print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()

if __name__ == '__main__':
    main()
