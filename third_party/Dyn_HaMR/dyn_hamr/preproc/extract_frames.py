import argparse
import imageio
import os
import subprocess


def split_frame(videopath,
    out_dir,
    fps=30,
    ext="jpg",
    down_scale=1,
    start_sec=0,
    end_sec=-1,
    overwrite=False,
    **kwargs,):
    import cv2
    base_name = videopath.replace('.mp4', '').replace('.MP4', '').replace('.avi', '')
    if os.path.isfile(videopath):
        print(f'processing {videopath} .....')
        # filepath = os.path.dirname(videopath)
        # videoname = os.path.basename(videopath)[:-4]
        if not os.path.exists(videopath):
            raise('{} not exists!'.format(videopath))

        vc = cv2.VideoCapture(videopath)

        path = out_dir
        if not os.path.exists(path):   
            os.makedirs(path)

        if vc.isOpened(): 
            rval , frame = vc.read()
            if '.MOV' not in videopath:
                cv2.imencode('.jpg',frame)[1].tofile(path +'/' + str(1).zfill(6) + '.jpg')
            else:
                cv2.imencode('.jpg',frame[::-1, ::-1])[1].tofile(path +'/' + str(0).zfill(6) + '.jpg')
        else:
            rval = False

        Frame = 2
        timeF = 1
        while rval:
            rval,frame = vc.read()     
            if rval==False:
                break

            if '.MOV' not in videopath:
                cv2.imencode('.jpg',frame)[1].tofile(path +'/' + str(Frame).zfill(6) + '.jpg')
            else:
                cv2.imencode('.jpg',frame[::-1, ::-1])[1].tofile(path +'/' + str(Frame).zfill(6) + '.jpg')

            Frame += 1
    elif os.path.isdir(videopath):
        print()
        print('moving ', base_name , 'to ', out_dir)
        os.system(f"cp -r {videopath} {out_dir}")
        raise ValueError
        return 0
    
    else:
        print(os.path.abspath(videopath), os.path.exists(videopath))
        raise ValueError(f"{videopath} does not exist or is not a valid file or directory.")

    return 0

def video_to_frames(
    path,
    out_dir,
    fps=30,
    ext="jpg",
    down_scale=1,
    start_sec=0,
    end_sec=-1,
    overwrite=False,
    **kwargs,
):
    """
    :param path
    :param out_dir
    :param fps
    :param down_scale (optional int)
    """
    path = str(path)
    base_name = path.replace('.mp4', '').replace('.MP4', '').replace('.avi', '')
    print(path, base_name)
    print(os.path.isfile(path), os.path.isdir(base_name))

    if os.path.isfile(path):
        os.makedirs(out_dir, exist_ok=True)

        arg_str = f"-copyts -qscale:v 2 -vf fps={fps}"
        if down_scale != 1:
            arg_str = f"{arg_str},scale='iw/{down_scale}:ih/{down_scale}'"
        if start_sec > 0:
            arg_str = f"{arg_str} -ss {start_sec}"
        if end_sec > start_sec:
            arg_str = f"{arg_str} -to {end_sec}"

        yn = "-y" if overwrite else "-n"
        cmd = f"ffmpeg -i {path} {arg_str} {out_dir}/%06d.{ext} {yn}"
        print(cmd)
    elif os.path.isdir(base_name):
        os.system(f"cp -r {base_name} {out_dir}")
        return 0
    else:
        print(path, os.path.exists(path))
        raise ValueError

    return subprocess.call(cmd, shell=True, stdin=subprocess.PIPE)

# def video_to_frames(
#     path,
#     out_dir,
#     fps=30,
#     ext="jpg",
#     down_scale=1,
#     start_sec=0,
#     end_sec=-1,
#     overwrite=False,
#     **kwargs,
# ):
#     """
#     :param path
#     :param out_dir
#     :param fps
#     :param down_scale (optional int)
#     """
#     path = str(path)
#     base_name = path.replace('.mp4', '').replace('.MP4', '').replace('.avi', '')
#     print(path, base_name)
#     print(os.path.isfile(path), os.path.isdir(base_name))

#     if os.path.isfile(path):
#         os.makedirs(out_dir, exist_ok=True)

#         arg_str = f"-copyts -qscale:v 2 -vf fps={fps}"
#         if down_scale != 1:
#             arg_str = f"{arg_str},scale='iw/{down_scale}:ih/{down_scale}'"
#         if start_sec > 0:
#             arg_str = f"{arg_str} -ss {start_sec}"
#         if end_sec > start_sec:
#             arg_str = f"{arg_str} -to {end_sec}"

#         yn = "-y" if overwrite else "-n"
#         # Changed %06d to start from 0 instead of 1
#         cmd = f"ffmpeg -i {path} {arg_str} -start_number 0 {out_dir}/%06d.{ext} {yn}"
#         print(cmd)
#     elif os.path.isdir(base_name):
#         os.system(f"cp -r {base_name} {out_dir}")
#         return 0
#     else:
#         print(path, os.path.exists(path))
#         raise ValueError

#     return subprocess.call(cmd, shell=True, stdin=subprocess.PIPE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="path to video")
    parser.add_argument(
        "--out_root", type=str, required=True, help="output dir for frames"
    )
    parser.add_argument(
        "--seqs",
        nargs="*",
        default=None,
        help="[optional] sequences to run, default runs all available",
    )
    parser.add_argument("--fps", type=int, default=30, help="fps to extract frames")
    parser.add_argument(
        "--ext", type=str, default="jpg", help="output filetype for frames"
    )
    parser.add_argument(
        "--down_scale", type=int, default=1, help="scale to extract frames"
    )
    parser.add_argument(
        "-ss", "--start_sec", type=float, default=0, help="seconds to start_sec"
    )
    parser.add_argument(
        "-es", "--end_sec", type=float, default=-1, help="seconds to end_sec"
    )
    parser.add_argument(
        "-y", "--overwrite", action="store_true", help="overwrite if already exist"
    )
    args = parser.parse_args()
    seqs_all = os.listdir(args.data_root)
    if args.seqs is None:
        args.seqs = seqs_all

    for seq in args.seqs:
        path = os.path.join(args.data_root, seq)
        print(f"EXTRACTING FRAMES FROM {path}")
        assert os.path.isfile(path)
        seq_name = os.path.splitext(os.path.basename(path.rstrip("/")))[0]
        out_dir = os.path.join(args.out_root, seq_name)
        video_to_frames(
            path,
            out_dir,
            args.fps,
            args.ext,
            args.down_scale,
            args.start_sec,
            args.end_sec,
            args.overwrite,
        )
