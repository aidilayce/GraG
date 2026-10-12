def propagate_in_video(predictor, session_id, frame_idx):
    """Propagate the session's prompts through the video; returns {frame_index: outputs}."""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=frame_idx,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame
