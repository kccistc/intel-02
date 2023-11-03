# Hand Written Equation Calculator
- Date : 2023.11.03

## Contributor
- Chungu, Gwon
  - training model
- Yeongdae, Kim
  - inferrence

## Project
- Goal : solution for hand written based equation

- ![EXAMPLE INPUT]("pictures/ex1.png")
- ![EXAMPLE OUTPUT]("pictures/ex2.png")

- Process : 
  - Training Model
    train based operations and digits datasets 
    - CNN based model
  - Inference

- Requirements
  - datasets
    - [Link] https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators/data
  - Reference
    - [Link] https://www.kaggle.com/code/rohankurdekar/handwritten-basic-math-equation-solver
    - 
  - Packages
  ```
  pip install numpy
  pip install pandas
  pip install opencv-python
  pip install tensorflow
  pip install plot 
  pip install imutils
  pip install scipy
  ```

  - To do list
    - interface camera (real time images)
    - more various operators
    - determinate that it is correct equation 
