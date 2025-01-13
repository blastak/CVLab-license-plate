# # 비디오 표시하기
# import gradio as gr
#
# def video_display(video_path):
#     return video_path
#
# demo = gr.Interface(video_display, gr.Video(), "video")
# demo.launch()
# # 성공, 녹화해서 처리하는 방식, 왜 좌우로 뒤집어지는지 모르겠음


# import gradio as gr
#
# def processing(img):
#     # return np.fliplr(im)
#     return img
#
# demo = gr.Interface(
#     processing,
#     gr.Image(streaming=True),
#     "image",
#     live=True
# )
# demo.launch()
# # 성공, 라이브로 output 가능, 근데 조금 느림



# import uuid
# import gradio as gr
# import cv2
# from time import sleep
#
# def v2i(video_path):
#     vc = cv2.VideoCapture(video_path)
#     width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     vwname = f"output_{uuid.uuid4()}.mp4"
#     # vw = cv2.VideoWriter(vwname, cv2.VideoWriter.fourcc(*'mp4v'), fps=vc.get(cv2.CAP_PROP_FPS), frameSize=(width, height))
#     while True:
#         succ, img = vc.read()
#         if not succ:
#             break
#         # TODO
#         # vw.write(img)
#         yield img
#         sleep(0.01)
#         # vw = cv2.VideoWriter(vwname, cv2.VideoWriter.fourcc(*'mp4v'), fps=vc.get(cv2.CAP_PROP_FPS), frameSize=(width, height))
#
# # demo = gr.Interface(v2i, gr.Video(), "video")
# demo = gr.Interface(v2i, gr.Video(), gr.Image(), live=True)
# demo.launch()
# # 성공, video로 녹화 image로 출력. 근데 버그가 좀 있음 ConnectionResetError: [WinError 10054] 현재 연결은 원격 호스트에 의해 강제로 끊겼습니다


# import numpy as np
# def sepia(input_img):
#     sepia_filter = np.array([
#         [0.393, 0.769, 0.189],
#         [0.349, 0.686, 0.168],
#         [0.272, 0.534, 0.131]
#     ])
#     sepia_img = input_img.dot(sepia_filter.T)
#     sepia_img /= sepia_img.max()
#     return sepia_img

import gradio as gr

def processing(img):
    return img

demo = gr.Interface(
    processing,
    gr.Image(streaming=True),
    "image",
    live=True
)
demo.launch()