# Unlearnable-Captcha
NTHUCS Machine Learning Final Project 

## Usage

### Training
Run `python main.py -t` to train model.

### Attack
Run `python main.py -a` to attack images.

### Attack
Run `python main.py -z 10` to generate 10 attacked images.

Implemented attack model:

* FGSM

* IFGSM

Adjust model name in `attack_model.attack()` in `unlearnable_captcha.attack()`, number of generate images in `tqdm` for-loop also in `unlearnable_captcha.attack()`.


### Arguments

`-p`: Specify proxy model (default: modelA)

`-n`: CAPTCHA classes number (default: 36 -> Uppercase characters + numbers)

`-l`: CAPTCHA length (default: 4 characters per CAPTCHA)

`-c`: CAPTCHA user customized string (default: None)

`-z`: CAPTCHA random attacked string 

## TO-do lists

### Bugs
* Resolve the problem of modifying size of CAPTCHA cannot train proxy model.

### Functions
* Let user input strings so that module can output specific CAPTCHA ground truth is the string.

* Construct more proxy model.

* Construct more attack model.

* Iterating attack until every alphabet is attacked sussceefully.