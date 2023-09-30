# IDM-TD3 Algorithm

This is the PyTorch implementation of Inverse Dynamic Model-Aware Twin Delayed Deep Deterministic Policy Gradient (IDM-TD3). Among them, the implementation of the kinematic part training refers to the original [TD3](https://github.com/sfujim/TD3) algorithm.

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym 0.22.0](https://github.com/openai/gym). Networks are trained using [PyTorch 1.12](https://github.com/pytorch/pytorch) and Python 3.10.

## Training Steps

Unlike other reinforcement learning algorithms, if you want to get a transferable IDM-TD3 model, you need to first train an IDM using supervised learning. For example, you can use the following command:

```
python dynamic.py --env_name HalfCheetah-v2
```

to test the **HalfCheetah-v2** environment in MuJoCo control task. The trained IDM is saved in "*./model/HalfCheetah-v2*" directory.



## Usage

