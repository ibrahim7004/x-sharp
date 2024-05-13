import cv2
import numpy as np

def estimate_noise(image):
    """Estimate the noise variance of an image."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    sigma = np.sum(np.sum(np.absolute(cv2.filter2D(image, -1, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w-2) * (h-2))
    return sigma

def remove_noise(image):
    """Apply Non-Local Means Denoising based on estimated noise level, reduced strength."""
    sigma_estimated = estimate_noise(image)
    h = 2.75 * sigma_estimated  # Reduced the multiplier for a lighter denoise
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    return denoised_image


def enhance_details(image):
    """Enhance details while preserving edges using a bilateral filter."""
    return cv2.bilateralFilter(image, d=13, sigmaColor=50, sigmaSpace=50)

def correct_warp(image, output_size=(256, 256)):
    """Correct perspective and scale issues in an image."""
    h, w = image.shape[:2]
    pts_src = np.float32([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]  # corners of original
    ])
    pts_dst = np.float32([
        [0, 0], [output_size[1] - 1, 0],
        [output_size[1] - 1, output_size[0] - 1], [0, output_size[0] - 1]  # stretched to new dimensions
    ])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped_image = cv2.warpPerspective(image, matrix, output_size)
    return warped_image

def adjust_white_balance(image):
    """Adjust white balance by scaling the channel intensities in the LAB color space."""
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 1] / 128))
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 2] / 128))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def adjust_contrast_brightness(image):
    """Adjust the contrast and brightness of the image using milder CLAHE settings."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_image = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

def inpaint_missing_regions(image, center=(190, 40), radius=27):
    """Inpaint the missing circular region more accurately."""
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted_image

def enhance_colors(image):
    """Increase saturation and adjust brightness for vibrancy."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.add(s, 150)  # Increase saturation boost
    v = cv2.add(v, 20)  # Increase brightness boost
    enhanced_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


def process_image(image_path):
    """Process an image through all steps."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None

    image = remove_noise(image)
    image = adjust_white_balance(image)
    image = enhance_details(image)
    image = correct_warp(image, output_size=(256, 256))  # Ensuring it fits the desired canvas size
    image = adjust_contrast_brightness(image)
    image = inpaint_missing_regions(image)
    image = enhance_colors(image)
    return image