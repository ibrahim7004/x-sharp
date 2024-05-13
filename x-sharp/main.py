import os
from image_enhancer import process_image
import cv2 

def main():
    input_dir = '/Users/zainnofal/Desktop/image_processing_files/xray_images'
    output_dir = 'processed_images/'
    os.makedirs(output_dir, exist_ok=True)
    
    images = [img for img in os.listdir(input_dir) if img.endswith('.jpg')]
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        processed_image = process_image(img_path)
        cv2.imwrite(os.path.join(output_dir, img_name), processed_image)

    # Call classify.py on the processed image
    os.system(f'python classify.py --data={output_dir} --model=classifier.model')

if __name__ == '__main__':
    main()
