# Gradient Descent
- Converge a model
- Use loss metrics to inform the model if its in the right direction
- Gemini Give me a small exmaple of gradient descrent, use ASCII characters to make plots
  - Gemini response:
    Okay, let's illustrate Gradient Descent with a simple example using ASCII plots. 

    Imagine we have a very simple model trying to predict a value (y) based on an input (x). Our "model" here will just be a straight line: `y = mx + c`.

    We want to find the best values for `m` (slope) and `c` (intercept) that make our line fit some data points as closely as possible. To measure how "close" our line is, we use a loss function. A common one is Mean Squared Error (MSE):

    `MSE = (1/N) * sum((y_actual - y_predicted)^2)`

    Gradient Descent is an iterative optimization algorithm that finds the minimum of a function. In our case, we want to minimize the MSE by adjusting `m` and `c`.

    **The Core Idea:**

    1.  **Start somewhere:** We begin with some initial, arbitrary values for `m` and `c`.
    2.  **Calculate the gradient:** We compute the gradient (the derivative) of the MSE with respect to `m` and `c`. The gradient tells us the direction of the steepest increase in the loss. To *decrease* the loss, we move in the *opposite* direction of the gradient.
    3.  **Take a step:** We update `m` and `c` by subtracting a fraction of their respective gradients. This fraction is called the **learning rate**.
    4.  **Repeat:** We repeat steps 2 and 3 until our loss stops decreasing significantly, meaning we've converged to a minimum.

    **Simple ASCII Example:**

    Let's say we want to fit a line to a single data point: `(x=2, y_actual=4)`.
    Our model is `y_predicted = m*x + c`. For simplicity, let's assume `c=0` and we only need to find the best `m`.

    Our loss function (MSE) for this single point is `(y_actual - m*x)^2 = (4 - m*2)^2`.

    We want to find `m` that minimizes this loss.

    **Iteration 1:**
    *   Let's start with `m = 0`.
    *   Loss = `(4 - 0*2)^2 = 16`.

        Plotting loss vs. m:
        `  L
        16 + . 
           | 
           | 
           +---- m`
           `0`

    *   Gradient of Loss w.r.t. `m`: `d/dm (4 - 2m)^2 = 2 * (4 - 2m) * (-2) = -4 * (4 - 2m)`
    *   At `m=0`, Gradient = `-4 * (4 - 0) = -16`.
    *   Learning Rate (alpha) = `0.1` (a small value).
    *   New `m` = `m - alpha * Gradient` = `0 - 0.1 * (-16)` = `0 + 1.6` = `1.6`.

    **Iteration 2:**
    *   Current `m = 1.6`.
    *   Loss = `(4 - 1.6*2)^2 = (4 - 3.2)^2 = 0.8^2 = 0.64`.

        Plotting loss vs. m:
        `  L
        16 + . 
         1 +   . 
        0.64+     .
           |      
           +--------- m`
           `0    1.6`

    *   Gradient at `m=1.6`: `-4 * (4 - 2*1.6) = -4 * (4 - 3.2) = -4 * 0.8 = -3.2`.
    *   New `m` = `1.6 - 0.1 * (-3.2)` = `1.6 + 0.32` = `1.92`.

    **Iteration 3:**
    *   Current `m = 1.92`.
    *   Loss = `(4 - 1.92*2)^2 = (4 - 3.84)^2 = 0.16^2 = 0.0256`.

        Plotting loss vs. m:
        `  L
        16 + . 
         1 +   . 
        0.64+     .
        0.0256+        .
           |         
           +------------ m`
           `0    1.6  1.92`

    *   Gradient at `m=1.92`: `-4 * (4 - 2*1.92) = -4 * (4 - 3.84) = -4 * 0.16 = -0.64`.
    *   New `m` = `1.92 - 0.1 * (-0.64)` = `1.92 + 0.064` = `1.984`.

    As you can see, `m` is getting closer and closer to `2`. If our actual data point was `(2, 4)`, the ideal `m` would be `2` (since `y = 2*x` would perfectly pass through `(2, 4)` if `c=0`). The loss would become `(4 - 2*2)^2 = 0`.

    **Visualizing the Loss Surface:**

    If we were to plot the Loss function `L(m) = (4 - 2m)^2` for different values of `m`, it would look like a parabola (a U-shape):

    ```
    Loss ^
         |
         |    /\ 
         |   /  \
         |  /    \
         | /------\
         +----------------> m
           0      2
    ```

    Gradient Descent is like a ball rolling down this parabola, always taking steps in the steepest downward direction until it reaches the bottom (the minimum loss).

    In more complex models with many parameters (like neural networks), this "loss surface" is multi-dimensional, but the principle remains the same: move in the direction opposite to the gradient to minimize the loss.
