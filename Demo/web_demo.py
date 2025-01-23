import os
import uuid

import cv2
import gradio as gr

SUBSAMPLE = 2


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

    while retval:
        if n_frames % SUBSAMPLE == 0:
            img = cv2.resize(img, (w_2, h_2))
            batch.append(img)
        if len(batch) == 2 * desired_fps:
            for b in batch:

                vw.write(b)

            vw.release()
            yield chunk_name

            batch = []
            chunk_name = os.path.join(output_path, f"output_{uuid.uuid4()}.mp4")
            vw = cv2.VideoWriter(chunk_name, fourcc, desired_fps, (w_2, h_2))

        retval, img = vc.read()
        n_frames += 1

    print('finish')


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
        with gr.Column():
            output_video = gr.Video(
                label="Processed Video", streaming=True, autoplay=True, loop=True
            )
            box = gr.Textbox(
                label='password for encryption',
                value='1q2w3e'
            )

    video.upload(
        fn=stream_object_detection,
        inputs=[video, box],
        outputs=[output_video],
    )

if __name__ == "__main__":
    demo.launch()
