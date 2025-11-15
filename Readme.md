# Project: Fuzzy System for Inverse Kinematics of a 2-Joint Planar Manipulator

## 1. Problem Definition

This project aims to solve the inverse kinematics problem of a 2-joint planar manipulator.  
The manipulator consists of two rigid links of lengths `l1 = 10` and `l2 = 7`, with joint angles `θ1` and `θ2`.  

**Objective:**  
Given a desired end-effector position `(x, y)` in the plane, compute the joint angles `θ1` and `θ2` that allow the manipulator to reach this position.  

**Challenges:**  
- The relationship between `(x, y)` and `(θ1, θ2)` is **non-linear**, and some positions `(x, y)` are **unreachable**.  
- Direct analytical computation can be difficult or impractical, especially for more complex manipulators.  

**Proposed Solution:**  
We will use an **adaptive fuzzy system** to approximate the inverse mapping `(x, y) → (θ1, θ2)` based on a dataset of reachable positions and corresponding joint angles.  
This approach allows handling non-linearity and uncertainty in the system effectively.
