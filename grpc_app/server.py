# grpc_app/server.py
import grpc
from concurrent import futures
import classify_pb2, classify_pb2_grpc
from infer import predict_bytes

import torch, os, resource, platform

if platform.system() == "Linux":
    try:
        resource.setrlimit(resource.RLIMIT_AS, (400 * 1024 * 1024, 400 * 1024 * 1024))
    except Exception as e:
        print(f"[WARN] setrlimit skipped: {e}")
else:
    print("[INFO] macOS detected — skip memory rlimit")

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

class ClassifierServicer(classify_pb2_grpc.ClassifierServicer):
    def Predict(self, request, context):
        topk = request.topk if request.topk > 0 else 5
        try:
            preds = predict_bytes(request.image, topk=topk)
            resp = classify_pb2.PredictionResponse(
                topk=[classify_pb2.Prediction(label=l, prob=p) for l, p in preds]
            )
            return resp
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return classify_pb2.PredictionResponse()

def serve(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    classify_pb2_grpc.add_ClassifierServicer_to_server(ClassifierServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"gRPC server started on port {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
