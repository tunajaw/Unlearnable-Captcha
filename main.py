import argparse
from unlearnable_captcha import unlearnable_captcha

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # cannot use "-h" argument since main.py use "-h" argument to show help messages  
    parser.add_argument("-s", "--size", nargs='+', required=False, type=int, default=[80, 30])
    parser.add_argument("-l", "--len", required=False, type=int, default=4)
    parser.add_argument("-n", "--nclass", required=False, type=int, default=36)
    parser.add_argument("-d", "--dataset", required=False, type=str, default='python_captcha')
    parser.add_argument("-c", "--customize", required=False, type=str, default=None) 
    parser.add_argument("-p", "--proxy_model", required=False, type=str, default='modelB')
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-a", "--attack", action="store_true")

    args = parser.parse_args()

    captcha = unlearnable_captcha(n_len=args.len, n_class=args.nclass, custom_string=args.customize)

    if(args.train):
        captcha.train(batch_size=128, dataset=args.dataset, model='modelB')
    else:
        captcha.load_proxy_model(model=args.proxy_model)

    if(args.attack):
        captcha.attack(attacked_model=['modelB'], method='FGSM', iter_atk=False, test_img_show=True)
    
    
