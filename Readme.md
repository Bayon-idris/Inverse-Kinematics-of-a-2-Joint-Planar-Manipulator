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
We use an **adaptive fuzzy system (ANFIS)** to approximate the inverse mapping  
\[
(x, y) \rightarrow (\theta_1, \theta_2)
\]  
based on a dataset of reachable positions and corresponding joint angles.  
This method handles non-linearity and uncertainty effectively.

---

## 2. Fuzzy System Design: Linguistic Variables and Membership Functions

To build the fuzzy inverse kinematics model, we first define the fuzzy variables involved in the system.  
This section follows the MATLAB experimentation guidelines provided in the coursework.

---

### 2.1 Input Variables

The system uses two input variables:
- **x**: End-effector horizontal position  
- **y**: End-effector vertical position  

Because the manipulator works only in the first quadrant, both inputs lie in the interval **[0, 19]**, exactly as defined in the experiment.

For each input variable, the experiment requires **three fuzzy sets**:
- *Low (L)*
- *Medium (M)*
- *High (H)*

This partitioning provides the necessary coverage of the workspace while keeping the rule base compact.

---

### 2.2 Output Variables

Two independent ANFIS models are constructed:
- **inv1 → θ1**
- **inv2 → θ2**

Since ANFIS supports only **one output per system**, building two parallel fuzzy systems is necessary to estimate both joint angles.

Each output uses **Sugeno-type rules**, where the rule conclusions are linear functions adjusted during training.

---

### 2.3 Membership Functions (MFs)

The experiment explicitly states that **three MFs must be used for each input**, resulting in:

- 3 membership functions for x  
- 3 membership functions for y  
→ **9 Takagi–Sugeno fuzzy rules per system**

This design ensures a complete 3×3 grid partition of the input space.

Common MF types used in ANFIS and allowed by the experiment include:
- Gaussian (`gaussmf`)
- Generalized bell (`gbellmf`)
- Triangular (`trimf`)

Although no specific shape is imposed, Gaussian or bell-shaped MFs are typically selected for smoother adaptation during training.

---

### 2.4 Summary of Step 2

| Component | Description |
|----------|-------------|
| Inputs | x, y |
| Input range | [0, 19] |
| Fuzzy sets per input | 3 (Low, Medium, High) |
| Total rules per system | 9 |
| Type | ANFIS (Sugeno) |
| Outputs | θ1 (first system), θ2 (second system) |

This structure forms the foundation for generating the fuzzy rule base and training the ANFIS models in the next steps.

## 3. Fuzzy Rule Base (Rules Definition)

The fuzzy rule base defines how the input variables (x, y) are mapped to the outputs (θ1, θ2) using fuzzy logic.  
Following the experiment, each system has **two inputs with three membership functions**, resulting in **9 fuzzy rules per system**.

---

### 3.1 Rule Base for θ1 (System inv1)

| Rule | IF x is | AND y is | THEN θ1 is (Sugeno linear output) |
|------|---------|-----------|----------------------------------|
| 1    | Low     | Low       | f1(x, y)                        |
| 2    | Low     | Medium    | f2(x, y)                        |
| 3    | Low     | High      | f3(x, y)                        |
| 4    | Medium  | Low       | f4(x, y)                        |
| 5    | Medium  | Medium    | f5(x, y)                        |
| 6    | Medium  | High      | f6(x, y)                        |
| 7    | High    | Low       | f7(x, y)                        |
| 8    | High    | Medium    | f8(x, y)                        |
| 9    | High    | High      | f9(x, y)                        |

> Each `fi(x, y)` is a linear function of the inputs, whose parameters are **automatically optimized** during ANFIS training.

---

### 3.2 Rule Base for θ2 (System inv2)

| Rule | IF x is | AND y is | THEN θ2 is (Sugeno linear output) |
|------|---------|-----------|----------------------------------|
| 1    | Low     | Low       | g1(x, y)                        |
| 2    | Low     | Medium    | g2(x, y)                        |
| 3    | Low     | High      | g3(x, y)                        |
| 4    | Medium  | Low       | g4(x, y)                        |
| 5    | Medium  | Medium    | g5(x, y)                        |
| 6    | Medium  | High      | g6(x, y)                        |
| 7    | High    | Low       | g7(x, y)                        |
| 8    | High    | Medium    | g8(x, y)                        |
| 9    | High    | High      | g9(x, y)                        |

> Each `gi(x, y)` is a linear function of the inputs, also trained by the ANFIS adaptive procedure.

---

### 3.3 Notes

- The **9 rules per system** fully cover all combinations of the 3×3 input fuzzy sets.  
- This rule base directly mirrors the MATLAB experiment described in the coursework.  
- During training, ANFIS automatically adjusts the parameters of both the membership functions and the linear functions in each rule to minimize the output error.


## 4. Fuzzy Inference and Defuzzification

After defining the fuzzy variables and rule base, the next step is to describe how the fuzzy system computes its outputs.  
This corresponds to the **inference and defuzzification** stages of fuzzy logic.

---

### 4.1 Fuzzy Inference

- The fuzzy inference system (FIS) evaluates **all rules simultaneously** for a given input `(x, y)`.
- For each rule:
  1. The **membership degree** of `x` and `y` to their respective fuzzy sets is computed.  
  2. The **firing strength** of the rule is calculated using the **AND operator** (commonly `min` or product).
- The contributions of all rules are then **aggregated** to produce a combined fuzzy output.

**In this project (ANFIS / Sugeno-type inference):**
- The output of each rule is a **linear function** of the inputs:  

## 5. Training and Testing

After designing the fuzzy system (variables, MFs, and rules), the next step is to **train the system** using the generated dataset and evaluate its performance.

---

### 5.1 Training Dataset

- The dataset contains **reachable positions** of the manipulator `(x, y)` and corresponding joint angles `(θ1, θ2)`.  
- Only points in the **first quadrant** and within the manipulator’s workspace are included.  
- Two datasets are used:
  - **Dataset for θ1** → training System inv1  
  - **Dataset for θ2** → training System inv2

**Note:** Each system is trained separately because ANFIS only supports a **single output** per model.

---

### 5.2 Training Procedure

1. Initialize the fuzzy system with:
   - 2 input variables (x, y)  
   - 3 membership functions per input  
   - 9 fuzzy rules per system
2. Apply the **ANFIS adaptive learning algorithm**:
   - Adjusts MF parameters and rule output coefficients  
   - Minimizes the **mean squared error (MSE)** between predicted θ and true θ values
3. Repeat the learning process for a fixed number of epochs or until convergence.

---

### 5.3 Testing and Evaluation

- After training, the system is tested on **new data points** not seen during training.
- Performance metrics:
  - **Prediction accuracy** of θ1 and θ2  
  - **Mean Squared Error (MSE)** between predicted and actual joint angles
- A **mapping surface plot** `(x, y) → θ` can be visualized to verify the system approximates the inverse kinematics correctly.
- The system can also identify **unreachable regions**, where predictions may be invalid (`NaN`) or outside physical constraints.

---

### 5.4 Notes

- This step closes the fuzzy system pipeline:  
