placeholder:
	@echo "Please run make download explicitly"

download:
	aws s3 cp --recursive s3://ray-serve-blog/composed-alpr-model/weights model_weights --acl=public-read

upload:
	aws s3 cp --recursive model_weights s3://ray-serve-blog/composed-alpr-model/weights --acl=public-read