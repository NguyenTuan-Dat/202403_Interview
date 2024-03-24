import os

if __name__ == '__main__':
    import argparse

    OUTPUT_E1 = "./outputs/OUTPUT_E1_FOR_E2.jpg"
    OUTPUT_E1 = os.path.abspath(OUTPUT_E1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", default="./inputs/raw.png")
    parser.add_argument("-o", "--output-path", default="./outputs/diffusion_raw.jpg")
    parser.add_argument("-p", "--prompt", default="Luna new year")
    parser.add_argument("-s", "--strength", default=0.5)
    args = parser.parse_args()

    os.system(f"python3 E1.py -i {args.image_path} -o {OUTPUT_E1}")
    os.chdir("./stable_diffusion.openvino")
    os.system(
        f'python3 demo.py --prompt "{args.prompt}" --init-image {args.image_path} --strength {args.strength} --mask {os.path.join(os.path.dirname(OUTPUT_E1), f"_mask_{os.path.basename(OUTPUT_E1)}")} --output {args.output_path}')
