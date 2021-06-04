

// Image Upload
const contentInpFile = document.getElementById("contentInputFile");
const styleInpFile = document.getElementById("styleInputFile");

// Image Upload Preview
const contentPreviewContainer = document.getElementById("contentImagePreview");
const stylePreviewContainer = document.getElementById("styleImagePreview");

const contentPreviewImage = contentPreviewContainer.querySelector(".preview_image");
const contentDefaultText = contentPreviewContainer.querySelector(".preview_default_text");

const stylePreviewImage = stylePreviewContainer.querySelector(".preview_image");
const styleDefaultText = stylePreviewContainer.querySelector(".preview_default_text");

function selectUploadFile(file, previewImage, previewDefaultText){
    if(file){
        previewImage.style.display = "block";
        previewDefaultText.style.display = "none";

        const reader = new FileReader();
        reader.addEventListener("load", function(){
            previewImage.setAttribute("src", this.result);

            data = new FormData();
            // data.append('model_name', model);
            // data.append('file', file);
        });
        reader.readAsDataURL(file);
    }
    else{
        console.log("Failed to load image");
        previewDefaultText.style.display = null;
        previewImage.style.display = null;
        previewImage.setAttribute("src", "");
    }
}

// contentInpFile.addEventListener("change", function(){selectUploadFile(contentPreviewImage, contentDefaultText)});
// styleInpFile.addEventListener("change", function(){selectUploadFile(stylePreviewImage, styleDefaultText)});
contentInpFile.addEventListener("change", function(){
    selectUploadFile(this.files[0], contentPreviewImage, contentDefaultText);
});
styleInpFile.addEventListener("change", function(){
    selectUploadFile(this.files[0], stylePreviewImage, styleDefaultText);
});