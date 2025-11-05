# grpc_app/client.py
import grpc
import classify_pb2, classify_pb2_grpc

def run(image_path: str, topk: int = 5, host: str = "127.0.0.1", port: int = 50051):
    with open(image_path, "rb") as f:
        data = f.read()
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = classify_pb2_grpc.ClassifierStub(channel)
    req = classify_pb2.ImageRequest(image=data, topk=topk)
    resp = stub.Predict(req)
    print("Top-K predictions:")
    for p in resp.topk:
        print(f"{p.label}: {p.prob:.4f}")

if __name__ == "__main__":
    # 示例：python client.py /path/to/cat.jpg
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "test.jpg")
