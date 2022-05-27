function varargout = phase1(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @phase1_OpeningFcn, ...
                   'gui_OutputFcn',  @phase1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before phase1 is made visible.
function phase1_OpeningFcn(hObject, ~, handles, varargin)

% Choose default command line output for phase1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes phase1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = phase1_OutputFcn(~, eventdata, handles) 

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in Upload.
function Upload_Callback(hObject, eventdata, handles)
global a
a=uigetfile('*.*');
a=imread(a);
axes(handles.axes1);
imshow(a);
setappdata(0,'a',a);


% --- Executes on button press in RESET.
function RESET_Callback(hObject, eventdata, handles)
global a
a=getappdata(0,'a');
imshow(a);

function undo_Callback(hObject, eventdata, handles)
global lst 
global a
imshow(lst);
a=lst;


% --- Executes on button press in redo.
function redo_Callback(hObject, eventdata, handles)
global redo 
global a
imshow(redo);
a=redo;

% --- Executes on button press in saveImg.
function saveImg_Callback(hObject, eventdata, handles)
global a
% saveas(gcf,'output.png');
imwrite(a, 'NewSavedImg.png');

% --- Executes on button press in gray_level.
function gray_level_Callback(hObject, eventdata, handles)
global a
a=rgb2gray(a);
imshow(a);

% --- Executes on button press in negative.
function negative_Callback(hObject, ~, handles)
global a
global lst 
lst = a;
global redo
L = 2 ^ 8;                       
neg = (L - 1) - a;
a = neg;
imshow(a);
redo = a;


% --- Executes on button press in log_transform.
function log_transform_Callback(hObject, ~, handles)
global a
global lst
lst = a;
global redo
gray_image = a;
double_value = im2double(gray_image);
out1= 2.5*log(1+double_value);
a = out1;
redo = a;
imshow(out1);



% --- Executes on button press in PowerLawTrans.
function PowerLawTrans_Callback(~, ~, ~)
global a
global lst
lst  =a;
global redo
c=1;
gray_image = a ;
double_value = im2double(gray_image);
out= c*(double_value.^1.5); 
a = out;
redo =a;
imshow(out);

% --- Executes on button press in Sub_Sambling.
function Sub_Sambling_Callback(hObject, eventdata, handles)
global a;
global lst
lst = a;
global redo
old_image= a;
sample_factor=10;

[r,c]=size(old_image);
new_image=old_image(1:sample_factor:r,1:sample_factor:c);
a = new_image;
redo =a;
imshow(new_image);



% --- Executes on button press in sharping.
function sharping_Callback(hObject, eventdata, handles)
global a;
global lst
lst=a;
global redo
gray_image = double(a);
[rows,cols]=size(gray_image);
mask = [0,1,0;
        1,-4,1;
        0,1,0];
out = gray_image;
for i=2:rows-1
 for j=2:cols-1
     
     temp = mask.*gray_image(i-1:i+1,j-1:j+1);
     value = sum(temp(:));
     out(i, j)= value;
end
end
out = uint8(out);
a =out;
redo = a;
imshow(out);


% --- Executes on button press in Adding.
function Adding_Callback(~, eventdata, ~)
global a;
global lst
lst = a;
global redo
prompt={'Enter Addition Value'};
dlgtitle = 'additionValue';
definput={'30'};
opts.Interpreter='tex';
AddVal = inputdlg(prompt,dlgtitle,[1 40],definput,opts);
AddVal = str2double(AddVal);
a = a + AddVal;
imshow(a);
redo = a;




% --- Executes on button press in subtration.
function subtration_Callback(hObject, eventdata, handles)
global a;
global lst
lst = a;
global redo
prompt={'Enter Subtraction Value'};
dlgtitle = 'Subtraction Value';
definput={'30'};
opts.Interpreter='tex';
SubtractValue = inputdlg(prompt,dlgtitle,[1 40],definput,opts);
SubtractValue = str2double(SubtractValue);
a = a - SubtractValue;
imshow(a);
redo = a;


% --- Executes on button press in Threshold.
function Threshold_Callback(hObject, eventdata, handles)
global a;
global lst
lst = a;
global redo
gray=a;
[ro,cl]=size(gray);
IM_BW=zeros(ro,cl);
for i=1:ro
    for j=1:cl
        if(gray(i,j)>150)
            IM_BW(i,j)=1;
        else
            IM_BW(i,j)=0;
        end
    end
end
a = IM_BW;
redo = a;
imshow(IM_BW);

% --- Executes on button press in Histogram.
function Histogram_Callback(hObject, eventdata, handles)
global a;
% gray_image=rgb2gray(a);
gray_image=a;
[rows,cols]=size(gray_image);
counts=zeros(1,256);
for i=1:rows
    for j=1:cols
        graylevel=gray_image(i,j);
        counts(graylevel+1)=counts(graylevel+1)+1;
    end
end
subplot(1,1,1)
graylevels = 0:255;
bar(graylevels , counts,'barwidth',1,'facecolor','b');
axes(handles.axes1);
imshow(graylevels);


% --- Executes on button press in Contrast.
function Contrast_Callback(hObject, eventdata, handles)
global a
global lst
lst = a;
global redo
gray= a;
w1=10;
w2=150;
r1=90;
r2=150;
l=255;
A=w1/r1;
b=(w2-w1)/(r2-r1);
g=(l-w2)/(l-r2);
[x,y,z]=size(gray);
for i=1:x 
    for j=1:y 
        if gray (i,j)<=r1
            r=gray(i,j);
        elseif gray(i,j)>r1 &&gray(i,j)<=150
            r=gray(i,j);
           gray(i,j)=(b*(r-r1))+w1;
        else
            r=gray(i,j);
            gray(i,j)=(g*(r-r2))+w2;
        end
    end
end
a=gray;
imshow(gray);
redo =a;

            
% --- Executes on button press in MeanFilter.
function MeanFilter_Callback(hObject, eventdata, handles)
global a
global lst
lst=a;
global redo;
img=a;
nImg=imnoise(img,'gaussian'); %"from documentation" adds zero-mean, Gaussian white noise with variance of 0.01 to grayscale image I
[m,n]=size(nImg);
%b=input('Enter Averaging Mask size: ');
b= 9; %Mask Size
z=ones(b);
[p,q]=size(z);
w=1:p;
x=round(median(w));
anz=zeros(m+2*(x-1),n+2*(x-1));
for i=x:(m+(x-1))
    for j=x:(n+(x-1))
        anz(i,j)=nImg(i-(x-1),j-(x-1));
    end
end
%**
sum=0;
x=0;
y=0;
for i=1:m
    for j=1:n
        for k=1:p
            for l=1:q
                sum= sum+anz(i+x,j+y)*z(k,l);
                y=y+1;
            end
            y=0;
            x=x+1;
        end
        x=0;
        res(i,j)=(1/(p*q))*(sum);
        sum=0;
    end
end
imshow(uint8(res));
a=uint8(res);
redo=a;


% --- Executes on button press in GaussianLowPass.
function GaussianLowPass_Callback(hObject, eventdata, handles)
global a
global lst
lst=a;
global redo;
input_image = a ;  
[M, N] = size(input_image);
FT_img = fft2(double(input_image));
D0 = 15; 
D0 = (D0^2)*2;
% Designing filter
u = 0:(M-1);
idx = find(u>M/2);
u(idx) = u(idx)-M;
v = 0:(N-1);
idy = find(v>N/2);
v(idy) = v(idy)-N;

% MATLAB library function meshgrid(v, u) returns
% 2D grid which contains the coordinates of vectors
% v and u. Matrix V with each row is a copy 
% of v, and matrix U with each column is a copy of u
[V, U] = meshgrid(v, u);
  
% Calculating Euclidean Distance
D = sqrt(U.^2+V.^2);

D = -D.^2;
% Comparing with the cut-off frequency and 
% determining the filtering mask
H = exp(D/D0);
  
% Convolution between the Fourier Transformed
% image and the mask
G = H.*FT_img;
  
% Getting the resultant image by Inverse Fourier Transform
% of the convoluted image using MATLAB library function 
% ifft2 (2D inverse fast fourier transform)  
output_image = real(ifft2(double(G)));
imshow(output_image, [ ]);
a=output_image;
redo=a;

% --- Executes on button press in ButterWorthLowPass.
function ButterWorthLowPass_Callback(hObject, eventdata, handles)
global a
global lst
lst=a;
global redo;
input_image = a;  
[M, N] = size(input_image);
FT_img = fft2(double(input_image));
D0 = 15; 
n=2*2;
% Designing filter
u = 0:(M-1);
idx = find(u>M/2);
u(idx) = u(idx)-M;
v = 0:(N-1);
idy = find(v>N/2);
v(idy) = v(idy)-N;

% MATLAB library function meshgrid(v, u) returns
% 2D grid which contains the coordinates of vectors
% v and u. Matrix V with each row is a copy 
% of v, and matrix U with each column is a copy of u
[V, U] = meshgrid(v, u);
  
% Calculating Euclidean Distance
D = sqrt(U.^2+V.^2);

D = D./ D0;
% Comparing with the cut-off frequency and 
% determining the filtering mask
H = 1./((1+D).^n);
  
% Convolution between the Fourier Transformed
% image and the mask
G = H.*FT_img;
  
% Getting the resultant image by Inverse Fourier Transform
% of the convoluted image using MATLAB library function 
% ifft2 (2D inverse fast fourier transform)  
output_image = real(ifft2(double(G)));
imshow(output_image, [ ]);
a=output_image;
redo=a;


% --- Executes on button press in Laplacianfilter.
function Laplacianfilter_Callback(hObject, eventdata, handles)
global a
global lst
lst=a;
global redo;
originalImage = a; 
gray_image = double(originalImage);
[rows,cols]=size(gray_image);
mask = [0,-1,0;-1,5,-1;0,-1,0];
out = gray_image;
for i=2:rows-1
 for j=2:cols-1
     temp = mask.*gray_image(i-1:i+1,j-1:j+1);
     value = sum(temp(:));
     out(i, j)= value;
end
end
out = uint8(out);
a=out;
redo=a;
imshow(out);


% --- Executes on button press in IdealLowPass.
function IdealLowPass_Callback(hObject, eventdata, handles)
% hObject    handle to IdealLowPass (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global a
global lst
lst=a;
global redo
input_image = a;
[M, N] = size(input_image);
FT_img = fft2(double(input_image));
D0 = 15; 
% Designing filter
u = 0:(M-1);
idx = find(u>M/2);
u(idx) = u(idx)-M;
v = 0:(N-1);
idy = find(v>N/2);
v(idy) = v(idy)-N;

% MATLAB library function meshgrid(v, u) returns
% 2D grid which contains the coordinates of vectors
% v and u. Matrix V with each row is a copy 
% of v, and matrix U with each column is a copy of u
[V, U] = meshgrid(v, u);
  
% Calculating Euclidean Distance
D = sqrt(U.^2+V.^2);
  
% Comparing with the cut-off frequency and 
% determining the filtering mask
H = double(D <= D0);
  
% Convolution between the Fourier Transformed
% image and the mask
G = H.*FT_img;
  
% Getting the resultant image by Inverse Fourier Transform
% of the convoluted image using MATLAB library function 
% ifft2 (2D inverse fast fourier transform)  
output_image = real(ifft2(double(G)));
 imshow(output_image, [ ]);
 a=output_image;
 redo=a;


% --- Executes on button press in SobolFilter.
function SobolFilter_Callback(hObject, eventdata, handles)
global a
global lst
lst=a;
global redo
originalImage = a;
gray_image = double(originalImage);
[rows,cols]=size(gray_image);
% mask = [-1 -2 -1;0 0 0;1 2 1];
mask = [-1 0 1;-2 0 2;-1 0 1];
out = gray_image;
for i=2:rows-1
 for j=2:cols-1
     temp = mask.*gray_image(i-1:i+1,j-1:j+1);
     value = sum(temp(:));
     out(i, j)= value;
end
end
out = uint8(out);
a=out;
redo=a;
imshow(out);



% --- Executes on button press in MedianFilter.
function MedianFilter_Callback(hObject, eventdata, handles)
global a
global lst
lst=a;
global redo
gray_image = a;    
[rows,cols]=size(gray_image);
out=gray_image;
for i=2:rows-1
 for j=2:cols-1
     temp = [gray_image(i-1, j-1) gray_image(i-1, j) gray_image(i-1, j + 1) gray_image(i, j-1) gray_image(i, j) gray_image(i, j + 1) gray_image(i + 1, j-1) gray_image(i + 1, j) gray_image(i + 1, j + 1)];
     temp = sort(temp);
     out(i, j)= temp(5);
end
end
imshow(out);
a=out;
redo=a;


% --- Executes on button press in UpSampling.
function UpSampling_Callback(hObject, eventdata, handles)
global a
global lst
lst=a;
global redo
image = a;
% [rows , cols , matricesNo] = size(image);
SamplingFactor = 9;
[g ,h] = size(a);
%g
%h
nrows=g*SamplingFactor;
ncols=h*SamplingFactor;
%nrows
%ncols
rcount=1;


for i=1:SamplingFactor:nrows
ccount=1;
    for c=1:SamplingFactor:ncols 
    outImage(i:(i+SamplingFactor-1),c:(c+SamplingFactor-1)) = image(rcount,ccount);
    ccount=ccount+1;
    end
    rcount=rcount+1;
end
imshow(outImage);
a=outImage;
redo=a;


% --- Executes on button press in inverseLog.
function inverseLog_Callback(hObject, eventdata, handles)
global a
global lst
lst = a;
global redo
gray_image = a;
double_value = im2double(gray_image);
out1= 0.5*log(1+double_value);
a = out1;
redo = a;
imshow(out1);

% --- Executes on button press in BitPlaneSlicing.
function BitPlaneSlicing_Callback(hObject, eventdata, handles)

global a
global lst
lst =a;
gray_image = a; 
figure;
subplot(2, 5, 1),
imshow(a);
[rows, cols] = size(gray_image);
newImage = zeros(rows,cols,8);
for k=1:8
    for row_index=1:1:rows
        for col_index=1:1:cols
            newImage(row_index,col_index,k)=bitget(gray_image(row_index,col_index),k);
        end
    end
subplot(2, 5, k+1),
imshow(newImage(:,:,k));
end


% --- Executes on button press in Exit.
function Exit_Callback(hObject, eventdata, handles)
closereq(); 
