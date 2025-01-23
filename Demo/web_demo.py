import os
import time
import uuid

import cv2
import gradio as gr
# import spaces

SUBSAMPLE = 2


# @spaces.GPU
def stream_object_detection(video, conf_threshold):
    cap = cv2.VideoCapture(video)

    video_codec = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    desired_fps = fps // SUBSAMPLE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

    iterating, frame = cap.read()

    n_frames = 0
    output_path = './streaming'
    os.makedirs(output_path, exist_ok=True)

    name = os.path.join(output_path, f"output_{uuid.uuid4()}.mp4")
    segment_file = cv2.VideoWriter(name, video_codec, desired_fps, (width, height))  # type: ignore
    batch = []

    while iterating:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if n_frames % SUBSAMPLE == 0:
            batch.append(frame)
        if len(batch) == 2 * desired_fps:

            print(f"starting batch of size {len(batch)}")
            start = time.time()

            for array in batch:
                # Draw a rectangle on each frame
                cv2.rectangle(array, (50, 200), (150, 300), (0, 255, 0), 2)
                frame = array[:, :, ::-1].copy()  # Convert RGB to BGR
                segment_file.write(frame)

            batch = []
            segment_file.release()
            yield name
            end = time.time()
            print("time taken for processing boxes", end - start)
            name = os.path.join(output_path, f"output_{uuid.uuid4()}.mp4")
            segment_file = cv2.VideoWriter(
                name, video_codec, desired_fps, (width, height)
            )  # type: ignore

        iterating, frame = cap.read()
        n_frames += 1


with gr.Blocks() as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    License plate swapping
    </h1>
    """
    )
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Video Source")
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.30,
            )
        with gr.Column():
            output_video = gr.Video(
                label="Processed Video", streaming=True, autoplay=True, loop=True
            )

    video.upload(
        fn=stream_object_detection,
        inputs=[video, conf_threshold],
        outputs=[output_video],
    )

if __name__ == "__main__":
    demo.launch()
