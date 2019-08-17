# DDPG-Tennis-ContinuousControl
Agent for solving the Unity "Tennis" Environment 

## Introduction
This is all about a Reinforcement Learning problem. Two agents play tennis. Each of them controls a racket to bounce the ball over the net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. This means they play not against each other.
To solve this environment the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). That means:
After each episode, the rewards that each agent received are added up. Then the maximum score is taken.

## Setting up the environment 
This project requires a python version 3.6 or higher.

Create a new python environment. Therefore you may follow the steps described here https://github.com/udacity/deep-reinforcement-learning#dependencies.

After creating the python environment the following packages must be installed:
 - torch 0.4.0
 - numpy 1.14.5
 - matplotlib 3.1.0
 - unityagents 0.3.0

#### How to install unityagents
An installation manual for Windows (contains also links for Linux - installation)  
https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installing-Unity-ml-agents-on-Windows

## The Environment
The environment is an adpated version of https://www.youtube.com/watch?v=IA2EcOPUNck and is a project of the Udacity Deep Reinforcement Learning nanodegree. The Tennis_Windows_x86_64.zip file offers this environment.
More informations: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis

## Getting started
Follow the instruction in the jupyter notebook Continuous_Control.ipynb.
It also contains a cell that discribes the action and state space.
