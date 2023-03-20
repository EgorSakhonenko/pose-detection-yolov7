import cv2
import torch
from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box
import argparse

# Path to the weights file
POSE_WEIGHTS = 'weights/yolov7-w6-pose.pt'

# Select the device to run the model
if torch.cuda.is_available():
    device = torch.device("cuda")

# elif torch.backends.mps.is_available():
#     device = torch.device("mps")

else:
    device = torch.device("cpu")
print('Selected Device : ', device)

# Load the pose estimation model
model = attempt_load(POSE_WEIGHTS, map_location=device)  # load model
model.eval()


# Function for changing modified coordinates to original
def change_coordinates(modified_coordinates, orig_frame, mod_frame):
    height, width, _ = orig_frame.shape
    mod_height, mod_width, _ = mod_frame.shape

    scale_x = width / mod_width
    scale_y = height / mod_height

    original_coordinates = modified_coordinates

    for i in range(original_coordinates.shape[0]):

        original_coordinates[i, 2] = round(modified_coordinates[i, 2] - (modified_coordinates[i, 4] / 2))
        original_coordinates[i, 3] = round(modified_coordinates[i, 3] - (modified_coordinates[i, 5] / 2))
        original_coordinates[i, 4] = round(modified_coordinates[i, 4] + original_coordinates[i, 2])
        original_coordinates[i, 5] = round(modified_coordinates[i, 5] + original_coordinates[i, 3])

        original_coordinates[i][2] *= scale_x
        original_coordinates[i][3] *= scale_y
        original_coordinates[i][4] *= scale_x
        original_coordinates[i][5] *= scale_y

        for j in range(8, original_coordinates.shape[1], 3):

            original_coordinates[i][j - 1] *= scale_x
            original_coordinates[i][j] *= scale_y

    return original_coordinates


# Function to perform pose estimation on a single frame
def pose_prediction(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # Convert the image from BGR to RGB
    img = letterbox(img, 960, stride=64, auto=True)[0]  # Resize and pad the image to a fixed size
    img_ = transforms.ToTensor()(img)                # Convert the image to a tensor and move it to the selected device
    img_ = img_.unsqueeze(0)
    img_ = img_.to(device)

    # Run the model on the image
    with torch.no_grad():
        output, _ = model(img_)

    output = non_max_suppression_kpt(output,
                                    conf_thres=0.25,
                                    iou_thres=0.65,
                                    nc=model.yaml['nc'],
                                    nkpt=model.yaml['nkpt'],
                                    kpt_label=True)

    # Convert the model output to a format that can be plotted on the image
    with torch.no_grad():
        output = output_to_keypoint(output)

    output = change_coordinates(output, frame, img)

    # Draw the skeleton and keypoints on the image
    for idx in range(output.shape[0]):

        plot_skeleton_kpts(frame, output[idx, 7:].T, 3)
        plot_one_box(output[idx, 2:6], frame, label='human', color=100, line_thickness=2)

    return frame


# Function for processing a video file frame by frame
def process_video(input_path, output_path):
    # Open the input video file and extract its properties
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize the output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame in the input video file
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret is True:
            processed_frame = pose_prediction(frame)
            processed_frame = cv2.resize(processed_frame, (width, height))
            out.write(processed_frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='Path to input video file')
    parser.add_argument('-o', type=str, required=True, help='Path to output video file')
    args = parser.parse_args()
    # Command line (terminal) input format: python3 pose_detection.py -i 'path to input file' -o 'path to output file'

    # inp = '/Users/egorsakhonenko/Downloads/test.mov'
    # outp = '/Users/egorsakhonenko/Downloads/result_test111.mp4'
    # process_video(inp, outp)

    process_video(args.i, args.o)


if __name__ == '__main__':
    main()