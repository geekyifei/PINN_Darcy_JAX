# PINN_Darcy_JAX
Implementations of PINN for 2D groundwater flow (Darcy) equation using JAX

- Forward mode: Train K DNN with data first and then solve the Darcy equation by training h DNN
- regression mode: Data-driven training of K and h DNNs sequentially
- inverse mode: Inverse estimation by solving the Darcy equation by training K and h DNNs jointly
