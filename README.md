# Playing MC with a Webcam Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Functionalities](#functionalities)
4. [Experiment](#experiment)
5. [Controls](#controls)
6. [Explanation](#explanation)

## Introduction

Welcome to the Playing MC with a Webcam project documentation. This project allows you to play Minecraft using a webcam thanks to computer vision and pydirectinput for changing actions that a user make on a webcam into action that a computer can read.

## Setup

To set up the project, follow these steps:

1. Install the 3.8.10 python version [https://www.python.org/downloads/release/python-3810/] (you might have to restart your machine) [why?](#explanation).
2. Clone this repository to your local machine.
3. In vscode to change the python version press `ctrl + shift + p` while having your python file opened and find `Python: Select Interpreter` once you find it click it and select the version you have just installed.
4. Install the needed packages `pip install -r requirements. txt`.
<div style="color:red">

**Warning:** On some devices you might have a problem closing the app.
In that case keep your nose in the reuqire are and use `alt + f4` or task manager to close it.

</div>

## Functionalities

The app provides the following functionalities using your webcam:

- **Moving**
- **Looking Around**
- **Jumping**
- **Placing Blocks**
- **Attacking**
- **Opening Inventory**

## Experiment

Since finishing the app I have tested it myself. Additionally I had a chance to show off this project on the open day at my university were a group of people tested the app as well. Thanks to these people I got a chance to see how it works on diffrent types of people and in diffrent clothing, which turned out to have big impact on how the model acted to the user actions. The main points that I got out of these tests are:

1. Wearing black clothes impact the ability of the model to read the moves of a user. For instance one of the people that tested the app was wearing full black pants. That caused the model to not pick up the controls for walking sometimes.
2. Having multiple people in the camera messes up with the model and his actions. For instance because the place that the people tested the app was in somewhat crouded when people walked behind the player in found the new target outside and went crazy for a while until it focused back on the player.
3. Having a bad enviroment or a lot of people in the camera impacts the time for an action to be made.

## Controls

- **Rasining you knees as if you are walking/running**: Move character action
- **Moving your nose in a direction outside the neutral zone (green square)**: Look around action
- **Jumping up**: Jump action
- **Moving your left arm the same way in-game character does**: Attacks/Mine action
- **Rasing your right hand above shoulders**: Placing blocks action
- **Moving your hands/hand between the left and right shoulders and hips point**: Opening Inventory/Single Left Click or Holding Right Click action

## Explanation

The reason why you have to install python version 3.8.10 is that the libary mediapipe that we use for fidning the pose of a person doesn't work on the latest python version. The best version that we can use this for is the version that we installed.
