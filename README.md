# Project Summary #
This bachelor thesis investigates how well different perception and control approaches work on a small-scale autonomous vehicle, the Duckiebot, inside the Duckietown ecosystem. 

It pursues two main objectives:

1. Object avoidance
    * Build and test a lightweight colour-based obstacle detector that spots the yellow rubber ducks used as obstacles in Duckietown.
    * Couple the detector with a Braitenberg-style reactive controller so that the robot can steer around the ducks without any global planning.

2. Lane keeping
Implement Deep Deterministic Policy Gradient (DDPG) and Twin Delayed DDPG (TD3) as deep-reinforcement-learning algorithms that can learn continuous steering and throttle commands from camera images alone:
    * Provide two independent implementations of each algorithm:
    a custom code base tailored to Duckietown (written in PyTorch and Gymnasium)
    the generic off-the-shelf implementation from the open-source framework Stable-Baselines3 (SB3).
    * Compare learning speed, final driving quality and robustness to unseen maps.

## Main results ##

### Object avoidance ###
* A simple HSV colour mask combined with a Braitenberg 2a controller lets the Duckiebot clear four out of five cluttered “duck fields” in the official Duckietown challenge.
* The average travel distance per field is 4.9 m; failures occur mainly when a duck sits exactly on the image centre, triggering symmetric motor commands.

### Lane keeping – custom implementation ###
* On the training map “loop_empty_simple” TD3 converges to an average episode reward of roughly 1350 and keeps a mean speed of about 0.8 m/s, outperforming the custom DDPG variant.
* On a harder map with tight left-right bends (“loop_empty”) custom TD3 still drives faster, but custom DDPG achieves higher total reward because TD3 occasionally loses stability.
* When the learnt policies are deployed on unseen layouts (4-way intersection and zig-zag track) the custom TD3 remains drivable and achieves the highest rewards among all tested agents, showing good generalisation.

### Lane keeping – stable baselines 3 implementation ###
* SB3 trains two to three times faster thanks to environment vectorisation.
* On the training map, SB3-TD3 reaches the highest reward (≈ 1410) but also shows mild over-steering artefacts.
* In unseen environments both SB3 agents suffer from pronounced overfitting: rewards drop sharply and the vehicles often oscillate or stop. The dependence on background colour (grass vs asphalt) is identified as a major cause.

## Overall conclusions ##
* TD3 is consistently superior to DDPG in both code bases.
* A hand-tuned, problem-specific implementation learns more slowly but transfers better to new situations than the generic SB3 version.
* Colour-thresholding plus Braitenberg control is a viable ultra-light solution for obstacle avoidance, but more sophisticated perception would be required for real-world deployment.

Future work should address domain-randomised training to close the sim-to-real gap, incorporate semantic segmentation to reduce texture bias, and explore state-dependent exploration noise to smooth remaining steering jitter.
