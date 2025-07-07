class CertificateGenerator {
    constructor() {
        this.canvas = document.getElementById('certificateCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.textInput = document.getElementById('certificateText');
        this.imageUpload = document.getElementById('imageUpload');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.cropBtn = document.getElementById('cropBtn');
        this.cropControls = document.getElementById('cropControls');
        this.cropButtons = document.getElementById('cropButtons');
        this.confirmCropBtn = document.getElementById('confirmCropBtn');
        this.cancelCropBtn = document.getElementById('cancelCropBtn');
        this.dragHandle = document.getElementById('dragHandle');
        this.charCounter = document.querySelector('.char-counter');
        
        this.backgroundImage = null;
        this.uploadedImage = null;
        this.originalImage = null; // Store original for cropping
        this.imagePosition = { x: 250, y: 200 }; // Center position
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        
        // Crop functionality (circular)
        this.isCropping = false;
        this.cropCenter = { x: 0, y: 0 };
        this.cropRadius = 0;
        this.cropStart = { x: 0, y: 0 };
        this.isSelectingCrop = false;
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.updateCharCounter();

        // Wait for the custom font to be loaded before drawing
        document.fonts.load('italic 400 1em "adobe-garamond-pro"').then(() => {
            console.log('Font loaded successfully!');
            // Now that font is loaded, load the background image and draw
            this.createBackgroundImage();
        }).catch(err => {
            console.error('Font failed to load, will use fallback.', err);
            // Draw anyway, the browser will use the fallback 'serif' font
            this.createBackgroundImage();
        });
    }
    
    createBackgroundImage() {
        // Load the diploma_modelo.png as background
        this.backgroundImage = new Image();
        this.backgroundImage.crossOrigin = 'anonymous'; // Allow cross-origin loading
        this.backgroundImage.onload = () => {
            // Adjust canvas size to match background image if needed
            this.canvas.width = this.backgroundImage.width;
            this.canvas.height = this.backgroundImage.height;
            
            // Calculate circle size to fit within the background decorative circle
            // Increased by additional 7% to 57% of the image width
            this.circleSize = Math.round(this.canvas.width * 0.57);
            
            // Update image position to center based on new canvas size
            this.imagePosition.x = (this.canvas.width - this.circleSize) / 2;
            this.imagePosition.y = (this.canvas.height - this.circleSize) / 2;
            
            this.redrawCertificate();
            // Enable download button once background is loaded
            this.downloadBtn.disabled = false;
        };
        this.backgroundImage.onerror = () => {
            console.error('Failed to load diploma_modelo.png');
            // Fallback to a simple background
            this.createFallbackBackground();
        };
        this.backgroundImage.src = 'diploma_modelo.png';
    }
    
    createFallbackBackground() {
        // Simple fallback if diploma_modelo.png fails to load
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(10, 10, this.canvas.width - 20, this.canvas.height - 20);
        this.redrawCertificate();
    }
    
    bindEvents() {
        this.textInput.addEventListener('input', () => {
            this.updateCharCounter();
            this.redrawCertificate();
        });
        
        this.imageUpload.addEventListener('change', (e) => {
            this.handleImageUpload(e);
        });
        
        this.canvas.addEventListener('mousedown', (e) => {
            this.handleMouseDown(e);
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            this.handleMouseMove(e);
        });
        
        this.canvas.addEventListener('mouseup', () => {
            this.handleMouseUp();
        });
        
        this.downloadBtn.addEventListener('click', () => {
            this.downloadCertificate();
        });
        
        this.cropBtn.addEventListener('click', () => {
            this.startCropping();
        });
        
        this.confirmCropBtn.addEventListener('click', () => {
            this.confirmCrop();
        });
        
        this.cancelCropBtn.addEventListener('click', () => {
            this.cancelCrop();
        });
    }
    
    updateCharCounter() {
        const length = this.textInput.value.length;
        this.charCounter.textContent = `${length}/80`;
        this.charCounter.style.color = length > 75 ? '#dc3545' : '#6c757d';
    }
    
    handleImageUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                this.originalImage = img; // Store original for cropping
                // Use dynamic circle size if available, otherwise default to 200
                const size = this.circleSize || 200;
                this.uploadedImage = this.resizeImage(img, size, size);
                this.redrawCertificate();
                this.updateDragHandle();
                this.downloadBtn.disabled = false;
                this.cropBtn.disabled = false;
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    resizeImage(img, targetWidth, targetHeight) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        
        // Calculate scale to fit image within target dimensions
        const scaleX = targetWidth / img.width;
        const scaleY = targetHeight / img.height;
        const scale = Math.min(scaleX, scaleY); // Use the smaller scale to ensure it fits
        
        const drawWidth = img.width * scale;
        const drawHeight = img.height * scale;
        
        // Center the image
        const offsetX = (targetWidth - drawWidth) / 2;
        const offsetY = (targetHeight - drawHeight) / 2;
        
        // Fill background with transparent
        ctx.clearRect(0, 0, targetWidth, targetHeight);
        
        // Draw the scaled image
        ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
        return canvas;
    }
    
    redrawCertificate() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw background image
        if (this.backgroundImage) {
            this.ctx.drawImage(this.backgroundImage, 0, 0);
        }
        
        // Draw uploaded image on top
        if (this.uploadedImage) {
            this.ctx.drawImage(
                this.uploadedImage,
                this.imagePosition.x,
                this.imagePosition.y
            );
        }
        
        // Draw custom text at bottom
        if (this.textInput.value.trim()) {
            this.ctx.fillStyle = '#2c3e50';
            this.ctx.font = 'italic 400 32px "adobe-garamond-pro", serif';
            this.ctx.textAlign = 'center';

            const text = this.textInput.value;
            const x = this.canvas.width / 2;
            const maxWidth = this.canvas.width * 0.7; // Max width for text block
            const lineHeight = 40; // Space between lines
            const lines = this._getWrappedLines(text, maxWidth);

            // Calculate starting Y to vertically center the block of text around the original position
            const blockCenterY = this.canvas.height * 0.8;
            const totalTextHeight = (lines.length - 1) * lineHeight;
            const startY = blockCenterY - totalTextHeight / 2;

            lines.forEach((line, index) => {
                this.ctx.fillText(line, x, startY + (index * lineHeight));
            });
        }
    }

    _getWrappedLines(text, maxWidth) {
        const words = text.split(' ');
        const lines = [];
        let currentLine = '';

        for (const word of words) {
            const testLine = currentLine ? `${currentLine} ${word}` : word;
            if (this.ctx.measureText(testLine).width > maxWidth && currentLine) {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        }
        if (currentLine) {
            lines.push(currentLine);
        }
        return lines;
    }

    updateDragHandle() {
        if (this.uploadedImage) {
            const rect = this.canvas.getBoundingClientRect();
            const scaleX = this.canvas.offsetWidth / this.canvas.width;
            const scaleY = this.canvas.offsetHeight / this.canvas.height;
            
            this.dragHandle.style.display = 'block';
            const size = this.circleSize || 200;
            this.dragHandle.style.left = `${rect.left + (this.imagePosition.x + size/2) * scaleX - 10}px`;
            this.dragHandle.style.top = `${rect.top + (this.imagePosition.y + size/2) * scaleY - 10}px`;
        }
    }
    
    handleMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / this.canvas.offsetWidth;
        const scaleY = this.canvas.height / this.canvas.offsetHeight;
        
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;
        
        if (this.isCropping) {
            // Handle circular crop selection
            this.isSelectingCrop = true;
            this.cropCenter.x = mouseX;
            this.cropCenter.y = mouseY;
            this.cropRadius = 0;
            return;
        }
        
        if (!this.uploadedImage) return;
        
        const size = this.circleSize || 200;
        if (mouseX >= this.imagePosition.x && 
            mouseX <= this.imagePosition.x + size &&
            mouseY >= this.imagePosition.y && 
            mouseY <= this.imagePosition.y + size) {
            
            this.isDragging = true;
            this.dragOffset.x = mouseX - this.imagePosition.x;
            this.dragOffset.y = mouseY - this.imagePosition.y;
            this.canvas.style.cursor = 'grabbing';
        }
    }
    
    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / this.canvas.offsetWidth;
        const scaleY = this.canvas.height / this.canvas.offsetHeight;
        
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;
        
        if (this.isCropping && this.isSelectingCrop) {
            // Update circular crop selection
            const dx = mouseX - this.cropCenter.x;
            const dy = mouseY - this.cropCenter.y;
            this.cropRadius = Math.sqrt(dx * dx + dy * dy);
            this.drawCropSelection();
            return;
        }
        
        if (!this.isDragging || !this.uploadedImage) return;
        
        this.imagePosition.x = mouseX - this.dragOffset.x;
        this.imagePosition.y = mouseY - this.dragOffset.y;
        
        const size = this.circleSize || 200;
        this.imagePosition.x = Math.max(0, Math.min(this.canvas.width - size, this.imagePosition.x));
        this.imagePosition.y = Math.max(0, Math.min(this.canvas.height - size, this.imagePosition.y));
        
        this.redrawCertificate();
        this.updateDragHandle();
    }
    
    handleMouseUp() {
        if (this.isCropping && this.isSelectingCrop) {
            this.isSelectingCrop = false;
            return;
        }
        
        this.isDragging = false;
        this.canvas.style.cursor = 'crosshair';
    }
    
    downloadCertificate() {
        try {
            // Check if canvas is tainted
            this.canvas.toDataURL();
            
            const link = document.createElement('a');
            link.download = 'certificate.png';
            link.href = this.canvas.toDataURL('image/png');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error('Download failed:', error);
            if (error.name === 'SecurityError') {
                alert('Download failed due to security restrictions. Please serve this page through a web server (not file://).');
            } else {
                alert('Download failed: ' + error.message);
            }
        }
    }
    
    startCropping() {
        if (!this.originalImage) return;
        
        this.isCropping = true;
        this.cropButtons.style.display = 'flex'; // Use flex for better layout
        this.canvas.style.cursor = 'crosshair';
        
        // Show original image for cropping
        this.showOriginalImageForCropping();
    }
    
    showOriginalImageForCropping() {
        // Clear canvas and show original image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw background
        if (this.backgroundImage) {
            this.ctx.drawImage(this.backgroundImage, 0, 0);
        }
        
        // Calculate dimensions to fit original image on canvas
        const maxWidth = this.canvas.width * 0.8;
        const maxHeight = this.canvas.height * 0.8;
        const aspectRatio = this.originalImage.width / this.originalImage.height;
        
        let displayWidth, displayHeight;
        if (aspectRatio > 1) {
            displayWidth = Math.min(maxWidth, this.originalImage.width);
            displayHeight = displayWidth / aspectRatio;
        } else {
            displayHeight = Math.min(maxHeight, this.originalImage.height);
            displayWidth = displayHeight * aspectRatio;
        }
        
        // Center the image
        this.cropImagePos = {
            x: (this.canvas.width - displayWidth) / 2,
            y: (this.canvas.height - displayHeight) / 2,
            width: displayWidth,
            height: displayHeight
        };
        
        // Draw original image
        this.ctx.drawImage(
            this.originalImage,
            this.cropImagePos.x,
            this.cropImagePos.y,
            this.cropImagePos.width,
            this.cropImagePos.height
        );
    }
    
    drawCropSelection() {
        // Redraw original image
        this.showOriginalImageForCropping();
        
        if (this.cropRadius <= 0) return;
        
        // Create a path for the circle
        this.ctx.save();
        
        // Draw dark overlay everywhere
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Create circular clipping path to reveal selected area
        this.ctx.globalCompositeOperation = 'destination-out';
        this.ctx.beginPath();
        this.ctx.arc(this.cropCenter.x, this.cropCenter.y, this.cropRadius, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Reset composite operation
        this.ctx.globalCompositeOperation = 'source-over';
        
        // Redraw the selected circular area
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.arc(this.cropCenter.x, this.cropCenter.y, this.cropRadius, 0, 2 * Math.PI);
        this.ctx.clip();
        
        // Calculate and draw the selected portion of the original image
        this.ctx.drawImage(
            this.originalImage,
            this.cropImagePos.x,
            this.cropImagePos.y,
            this.cropImagePos.width,
            this.cropImagePos.height
        );
        
        this.ctx.restore();
        
        // Draw selection border circle
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([8, 8]);
        this.ctx.beginPath();
        this.ctx.arc(this.cropCenter.x, this.cropCenter.y, this.cropRadius, 0, 2 * Math.PI);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        this.ctx.restore();
    }
    
    confirmCrop() {
        if (this.cropRadius <= 0) {
            alert('Please select an area to crop.');
            return;
        }
        
        // Calculate source coordinates relative to original image
        const scaleX = this.originalImage.width / this.cropImagePos.width;
        const scaleY = this.originalImage.height / this.cropImagePos.height;
        
        // Convert display coordinates to source coordinates
        const sourceCenterX = (this.cropCenter.x - this.cropImagePos.x) * scaleX;
        const sourceCenterY = (this.cropCenter.y - this.cropImagePos.y) * scaleY;
        const sourceRadius = this.cropRadius * scaleX; // Use scaleX for consistent scaling
        
        // Ensure the crop area is within image bounds
        const maxRadius = Math.min(
            sourceCenterX, 
            sourceCenterY, 
            this.originalImage.width - sourceCenterX, 
            this.originalImage.height - sourceCenterY
        );
        const finalRadius = Math.min(sourceRadius, maxRadius);
        
        // Calculate crop area bounds
        const cropSize = finalRadius * 2;
        const cropX = sourceCenterX - finalRadius;
        const cropY = sourceCenterY - finalRadius;
        
        // Create circular cropped image - make it square
        const cropCanvas = document.createElement('canvas');
        const cropCtx = cropCanvas.getContext('2d');
        
        cropCanvas.width = cropSize;
        cropCanvas.height = cropSize;
        
        // Fill with transparent background first
        cropCtx.clearRect(0, 0, cropSize, cropSize);
        
        // Create circular clipping path
        cropCtx.save();
        cropCtx.beginPath();
        cropCtx.arc(finalRadius, finalRadius, finalRadius, 0, 2 * Math.PI);
        cropCtx.clip();
        
        // Draw the image portion within the circle
        cropCtx.drawImage(
            this.originalImage,
            cropX, cropY, cropSize, cropSize,
            0, 0, cropSize, cropSize
        );
        
        cropCtx.restore();
        
        // Create new image from cropped canvas
        const croppedImg = new Image();
        croppedImg.onload = () => {
            // Use the calculated circle size to match background
            const size = this.circleSize || 200;
            this.uploadedImage = this.resizeImageCircular(croppedImg, size, size);
            this.cancelCrop();
            this.redrawCertificate();
            this.updateDragHandle();
        };
        croppedImg.src = cropCanvas.toDataURL();
    }
    
    resizeImageCircular(img, targetWidth, targetHeight) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        
        // Clear with transparent background
        ctx.clearRect(0, 0, targetWidth, targetHeight);
        
        // Create circular clipping path
        ctx.save();
        ctx.beginPath();
        ctx.arc(targetWidth/2, targetHeight/2, Math.min(targetWidth, targetHeight)/2, 0, 2 * Math.PI);
        ctx.clip();
        
        // Draw the image to fill the circle
        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
        
        ctx.restore();
        
        return canvas;
    }
    
    cancelCrop() {
        this.isCropping = false;
        this.isSelectingCrop = false;
        this.cropButtons.style.display = 'none';
        this.canvas.style.cursor = 'default';
        this.redrawCertificate();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new CertificateGenerator();
});