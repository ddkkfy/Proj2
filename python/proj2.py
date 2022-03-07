import torch
import time

from python.sgxutils import SGXUtils

def main(args):
    sgxutils = SGXUtils()

    method_time = 0.0
    native_time = 0.0
    for i in range(0, 5):
        l = torch.nn.Linear(args.in_features, args.out_features, bias=False).cuda()
        x = torch.randn(args.batch, args.in_features).cuda()


        t1 = time.process_time()
        # given the weight; precompute w * r
        sgxutils.precompute(l.weight, args.batch)

        # x_blinded = x + r
        x_blinded = sgxutils.addNoise(x)

        # y_blinded = w * x_blinded
        y_blinded = l(x_blinded)

        # y_recovered = y_blinded - w * r
        y_recovered = sgxutils.removeNoise(y_blinded)
        method = time.process_time() - t1
        t2 = time.process_time()
        s = sgxutils.nativeMatMul(l.weight, x)
        native = time.process_time() - t2

        y_expected = l(x)

        print("Total diffs:", abs(y_expected - y_recovered).sum())
        print("Native sum is: ", s.sum())
        #print("Total native diffs:", abs(s - y_expected.cpu()).sum())
        #print("Total inner Enclave diffs:", abs(s - y_recovered.cpu()).sum())
        method_time = method_time + method
        native_time = native_time + native

    #print("\n")
    print(f"The speedup is {round((native_time/method_time), 3)}x")
    return 0
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_features', type=int, default=10, help="Input feature of Linear layer")
    parser.add_argument('--out_features', type=int, default=30,  help="Output feature of Linear layer")
    parser.add_argument('--batch', type=int, default=5, help="Input batch size")

    args = parser.parse_args()
    main(args)