
STEPS TO USE FLUX-IMG-GEN

install.bat will attempt to create a venv (folder that will contain python dependnecies) and install the required dependencies


you'll need to download and install cuda 11.8 from nvidia


edit flux_img_gen.py directly to change image generation prompt and parameters
	there are several things you can edit, but the g_clip_prompt is the only one you need to at at first
	loras are supported, add the path in the script as indicated in the #comments
	

run launch_flux-img-gen.bat to generate images
	the first run will take a while as it downloads 30gb to your profile cache (it will remain there until you delete it)
	this script runs the 16 bit model (higher quality) and will be slower
	
