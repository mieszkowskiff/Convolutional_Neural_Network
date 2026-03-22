# Adversarial attack

If you want to read more about adversarial attacks [read here](https://en.wikipedia.org/wiki/Adversarial_machine_learning).
To run adversarial attack select model name (from the `./models` directory) in `adversarial_attack.py`, select `picture_index` and run the program.

## Results

Before attack            |  Afer attack
:-------------------------:|:-------------------------:
![Before](./../../img/before_adversarial_attack.png)  |  ![After](./../../img/after_adversarial_attack.png)

Even though for human picutres above seems the same, model is confused:
```{bash}
Loss without attack: 0.06967619806528091
Prediction without attack: car
Loss with attack: 1.4984451532363892
Prediction with attack: truck
```

## Creating adversarial dataset

To create adversarial dataset select model name (from the `./models` directory) in `adversarial_datasetpy` and run the program.