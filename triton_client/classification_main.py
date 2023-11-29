import cv2
from pathlib import Path


from grpc_image_client import *

class Video_stream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            return frame

class Info:
    def __init__(self, url, model_name, model_version, input, ) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     action="store_true",
    #     required=False,
    #     default=False,
    #     help="Enable verbose output",
    # )
    # parser.add_argument(
    #     "-a",
    #     "--async",
    #     dest="async_set",
    #     action="store_true",
    #     required=False,
    #     default=False,
    #     help="Use asynchronous inference API",
    # )
    # parser.add_argument(
    #     "--streaming",
    #     action="store_true",
    #     required=False,
    #     default=False,
    #     help="Use streaming inference API",
    # )
    # parser.add_argument(
    #     "-m", "--model-name", type=str, required=True, help="Name of model"
    # )
    # parser.add_argument(
    #     "-x",
    #     "--model-version",
    #     type=str,
    #     required=False,
    #     default="",
    #     help="Version of model. Default is to use latest version.",
    # )
    # parser.add_argument(
    #     "-b",
    #     "--batch-size",
    #     type=int,
    #     required=False,
    #     default=1,
    #     help="Batch size. Default is 1.",
    # )
    # parser.add_argument(
    #     "-c",
    #     "--classes",
    #     type=int,
    #     required=False,
    #     default=1,
    #     help="Number of class results to report. Default is 1.",
    # )
    # parser.add_argument(
    #     "-s",
    #     "--scaling",
    #     type=str,
    #     choices=["NONE", "INCEPTION", "VGG"],
    #     required=False,
    #     default="NONE",
    #     help="Type of scaling to apply to image pixels. Default is NONE.",
    # )
    # parser.add_argument(
    #     "-u",
    #     "--url",
    #     type=str,
    #     required=False,
    #     default="localhost:8001",
    #     help="Inference server URL. Default is localhost:8001.",
    # )
    # parser.add_argument(
    #     "image_filename",
    #     type=str,
    #     nargs="?",
    #     default=None,
    #     help="Input image / Input folder.",
    # )
    FLAGS = parser.parse_args()

    FLAGS.url = "192.168.191.129:8001"
    FLAGS.model_name = "trash_classification"
    FLAGS.streaming = True
    FLAGS.classes = 4
    FLAGS.batch_size = 1
    FLAGS.model_version = ""
    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    metadata_request = service_pb2.ModelMetadataRequest(
        name=FLAGS.model_name, version=FLAGS.model_version
    )
    metadata_response = grpc_stub.ModelMetadata(metadata_request)

    config_request = service_pb2.ModelConfigRequest(
        name=FLAGS.model_name, version=FLAGS.model_version
    )
    config_response = grpc_stub.ModelConfig(config_request)

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
        metadata_response, config_response.config
    )

    supports_batching = max_batch_size > 0
    if not supports_batching and FLAGS.batch_size != 1:
        raise Exception("This model doesn't support batching.")

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []

    # Send request
    if FLAGS.streaming:
        for response in grpc_stub.ModelStreamInfer(
            requestGenerator(
                input_name,
                output_name,
                c,
                h,
                w,
                format,
                dtype,
                FLAGS,
                result_filenames,
                supports_batching,
            )
        ):
            responses.append(response)
    else:
        for request in requestGenerator(
            input_name,
            output_name,
            c,
            h,
            w,
            format,
            dtype,
            FLAGS,
            result_filenames,
            supports_batching,
        ):
            if not FLAGS.async_set:
                responses.append(grpc_stub.ModelInfer(request))
            else:
                requests.append(grpc_stub.ModelInfer.future(request))

    # For async, retrieve results according to the send order
    if FLAGS.async_set:
        for request in requests:
            responses.append(request.result())

    error_found = False
    idx = 0
    for response in responses:
        if FLAGS.streaming:
            if response.error_message != "":
                error_found = True
                print(response.error_message)
            else:
                postprocess(
                    response.infer_response,
                    result_filenames[idx],
                    FLAGS.batch_size,
                    supports_batching,
                )
        else:
            postprocess(
                response, result_filenames[idx], FLAGS.batch_size, supports_batching
            )
        idx += 1

    if error_found:
        sys.exit(1)

    print("PASS")