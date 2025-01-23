import os
import uuid

import cv2
import gradio as gr

from Utils import get_pretty_datetime

from inferencing import Demo_Runner

SUBSAMPLE = 2

demo_runner = Demo_Runner()

def stream_object_detection(video, password):
    vc = cv2.VideoCapture(video)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore
    fps = int(vc.get(cv2.CAP_PROP_FPS))

    desired_fps = fps // SUBSAMPLE
    w_2 = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    h_2 = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

    retval, img = vc.read()

    n_frames = 0
    output_path = './streaming'
    os.makedirs(output_path, exist_ok=True)

    batch = []
    chunk_name = os.path.join(output_path, f"output_{uuid.uuid4()}.mp4")
    vw = cv2.VideoWriter(chunk_name, fourcc, desired_fps, (w_2, h_2))

    demo_runner.setup()

    while retval:
        if n_frames % SUBSAMPLE == 0:
            img = cv2.resize(img, (w_2, h_2))
            batch.append(img)
        if len(batch) == 2 * desired_fps:
            for b in batch:
                img2, img3 = demo_runner.loop(b, password, password)
                vw.write(img2)

            vw.release()
            yield chunk_name

            batch = []
            chunk_name = os.path.join(output_path, f"output_{uuid.uuid4()}.mp4")
            vw = cv2.VideoWriter(chunk_name, fourcc, desired_fps, (w_2, h_2))

        retval, img = vc.read()
        n_frames += 1

    print(get_pretty_datetime(), password)


with gr.Blocks() as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    License Plate De-identification (Encryption)
    </h1>
    """
    )
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Video Source")
            box = gr.Textbox(label='password for encryption', value='1q2w3e')
        with gr.Column():
            output_video = gr.Video(
                label="Encrypted Video", streaming=True, autoplay=True, loop=True
            )

    video.upload(
        fn=stream_object_detection,
        inputs=[video, box],
        outputs=[output_video]
    )

if __name__ == "__main__":
    demo.launch()
