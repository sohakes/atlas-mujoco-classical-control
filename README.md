This repository is a work-in-progress implementation of the paper Generalized biped walking control (https://dl.acm.org/citation.cfm?id=1781156) for the Atlas robot.

The algorithm runs, but it does not work as expected yet. For the program to run, it is necessary to replace './assets/atlas_v5.xml' for the correct path of the Atlas mujoco xml file.

The inverse kinematics algorithms is implemented from the book Introduction to Humanoid Robotics (https://www.springer.com/gp/book/9783642545351 section 2.5.8), with the joint range limit restriction added from the book Autonomous Robots (https://www.springer.com/gp/book/9780387095370 section 2.4.1.1).

The gravity compensation algorithm (which works correctly using torque control) and the inverted pendulum model were available in the original paper.

Forward kinematics was not implemented since Mujoco already have it.

The IK only works if directly changing the joint positions (in qpos, the joint coordinates vector) instead of using the PID. It's cheating, but the IK algorithm is probably correct. There is also a problem with Mujoco's forward kinematics, I think it's related to me not resetting qpos and qvel correctly, and so it gets to absurdly high values that causes some instability when using PID control. As a next step, I will implement the forward kinematics myself to see if it solves the problem.