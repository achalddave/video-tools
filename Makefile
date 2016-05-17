PROTOC=protoc

.PHONY : all clean

all: video_frames_pb2.py video_frames_pb.lua

video_frames_pb2.py: video_frames.proto
	$(PROTOC) video_frames.proto --python_out .

video_frames_pb.lua: video_frames.proto
	$(PROTOC) video_frames.proto --lua_out .

clean:
	rm *_pb2.py
	rm *_pb.lua
