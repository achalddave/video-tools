PROTOC=protoc

.PHONY : all clean

all: util/video_frames_pb2.py util/video_frames_pb.lua

util/video_frames_pb2.py: util/video_frames.proto
	$(PROTOC) util/video_frames.proto --python_out .

util/video_frames_pb.lua: util/video_frames.proto
	$(PROTOC) util/video_frames.proto --lua_out .

clean:
	rm util/*_pb2.py
	rm util/*_pb.lua
