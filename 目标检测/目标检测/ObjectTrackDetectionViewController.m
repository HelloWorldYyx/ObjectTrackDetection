//
//  ObjectTrackDetectionViewController.m
//  案列
//
//  Created by admin on 21/3/18.
//  Copyright © 2018年 Yanyx. All rights reserved.
//

#import "ObjectTrackDetectionViewController.h"
#import "objectTracking.h"
#import <Vision/Vision.h>
#import <AVFoundation/AVFoundation.h>
#import "UIImage+Rotate.h"
#define SCREENHEIGHT [UIScreen mainScreen].bounds.size.height
#define SCREENWIDTH [UIScreen mainScreen].bounds.size.width
#define mixNume 0.3
#define nms_threshold 0.45

typedef void(^PropertyChangeBlock)(AVCaptureDevice *captureDevice);
struct Prediction {
    NSInteger labelIndex;
    CGFloat confidence;
    CGRect boundingBox;
};
typedef struct Prediction Prediction;
@interface ObjectTrackDetectionViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate,AVCaptureAudioDataOutputSampleBufferDelegate>
{
    // 会话
    AVCaptureSession            *captureSession_;
    AVCaptureDevice             *captureDevice;
    AVCaptureDeviceInput        *videoCaptureDeviceInput;
    AVCaptureDeviceInput        *audioCaptureDeviceInput;
    AVCaptureVideoDataOutput    *captureVideoDataOutput;
    AVCaptureAudioDataOutput    *captureAudioDataOutput;
    dispatch_queue_t            myQueue;
    AVCaptureConnection         *videoConnection;
    AVCaptureConnection         *audioConnection;
    AVCaptureVideoPreviewLayer  *captureVideoPreviewLayer;
    
    BOOL isDetection; //是否检测
    NSInteger detectionNum; //检测次数
    CGPoint beginPoint;
    CGPoint endPoint;
}

@property (nonatomic, strong) UILabel *infoLabel;
@property (nonatomic, strong) UIView *boxView;

@property (nonatomic, retain) VNSequenceRequestHandler *sequenceHandler;
@property (nonatomic, strong) VNDetectedObjectObservation *lasetObservation;

@end

@implementation ObjectTrackDetectionViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
  
    isDetection = YES;
    detectionNum = 0;
    self.view.backgroundColor = [UIColor whiteColor];
    self.navigationItem.title = @"目标跟踪检测";
    [self createCamera];
    [self.view addSubview:self.boxView];
    [self.view addSubview:self.infoLabel];
}


- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    self.lasetObservation = nil;
    
    UITouch *touch = touches.anyObject;
    beginPoint = [touch locationInView:self.view];
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    UITouch *touch = touches.anyObject;
    endPoint = [touch locationInView:self.view];
    
    double x , y;
    if (beginPoint.x < endPoint.x) {
        x = beginPoint.x;
    } else {
        x = endPoint.x;
    }
    
    if (beginPoint.y < endPoint.y) {
        y = beginPoint.y;
    } else {
        y = endPoint.y;
    }
    
    self.boxView.frame = CGRectMake(x, y, fabs(beginPoint.x - endPoint.x), fabs(beginPoint.y - endPoint.y));
    
    
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    UITouch *touch = touches.anyObject;
    endPoint = [touch locationInView:self.view];
    CGRect convertedRect = [captureVideoPreviewLayer metadataOutputRectOfInterestForRect:self.boxView.frame];
    convertedRect = CGRectMake(convertedRect.origin.x , convertedRect.origin.y, convertedRect.size.width, convertedRect.size.height);
    VNDetectedObjectObservation * newObservation = [VNDetectedObjectObservation observationWithBoundingBox:convertedRect];
    self.lasetObservation = newObservation;
    _sequenceHandler = nil;
    
    [self changeFocusWithPoint:CGPointZero];
    
}


- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection{
    if (output == captureVideoDataOutput) {
        if (isDetection) {
            [self objectDetectionWithSampleBuffer:sampleBuffer];
        } else {
            [self objectTrackingWithSamplebuffer:sampleBuffer];
        }
        
    }
}

- (void)objectDetectionWithSampleBuffer:(CMSampleBufferRef)sampleBuffer{
    UIImage *newPhoto = [self imageFromSampleBuffer:sampleBuffer];
    newPhoto = [UIImage imageWithCGImage:newPhoto.CGImage scale:1.0 orientation:UIImageOrientationRight];
    newPhoto = [newPhoto fixOrientation];
    
    
    objectTracking *model = [[objectTracking alloc] init];
    CGFloat width  = SCREENWIDTH;
    CGFloat height = SCREENHEIGHT;
    NSDictionary *classsesDic = model.model.modelDescription.metadata[MLModelCreatorDefinedKey];
    VNCoreMLModel *coreMLModel = [VNCoreMLModel modelForMLModel:model.model error:nil];
    VNCoreMLRequest *request = [[VNCoreMLRequest alloc] initWithModel:coreMLModel completionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable error) {
        CGFloat confidenceThreshold = mixNume;
        NSMutableArray *unorderedPredictions = [NSMutableArray array];
        VNCoreMLFeatureValueObservation *classeObservation  = request.results[0];
        VNCoreMLFeatureValueObservation *boxRectObservation = request.results[1];
        
        MLMultiArray *coordinates = classeObservation.featureValue.multiArrayValue;
        MLMultiArray *confidence  = boxRectObservation.featureValue.multiArrayValue;
        
        int numBoundingBoxes = confidence.shape[0].intValue;
        int numClasses = confidence.shape[1].intValue;
        for (int b = 0; b < numBoundingBoxes; b++) {
            float maxConfidence = 0.0;
            int maxIndex = 0;
            for (int c = 0; c < numClasses; c++) {
                NSNumber *confNumber = confidence[b * numClasses + c];
                double conf = [confNumber doubleValue];
                if (conf > maxConfidence) {
                    maxConfidence = conf;
                    maxIndex = c;
                }
            }
            if (maxConfidence > confidenceThreshold) {
                NSNumber *xNumber = coordinates[b * 4];
                NSNumber *yNumber = coordinates[b * 4 + 1];
                NSNumber *wNumber = coordinates[b * 4 + 2];
                NSNumber *hNumber = coordinates[b * 4 + 3];
                double x = [xNumber doubleValue];
                double y = [yNumber doubleValue];
                double w = [wNumber doubleValue];
                double h = [hNumber doubleValue];
                
                CGRect rect = CGRectMake(x - w/2, y - h/2, w, h);
                Prediction prediction = {maxIndex, maxConfidence,rect};
                NSValue *predictionValue = [NSValue valueWithBytes:&prediction objCType:@encode(Prediction)];
                [unorderedPredictions addObject:predictionValue];
            }
        }
        if (unorderedPredictions.count <=0) {
            dispatch_async(dispatch_get_main_queue(), ^{
                AVCaptureDevice * camera = videoCaptureDeviceInput.device;
                if ([camera lockForConfiguration:nil]) {
                    [camera setExposurePointOfInterest:CGPointMake(0.5, 0.5)];
                    [camera setExposureMode:(AVCaptureExposureMode)AVCaptureExposureModeAutoExpose];
                }
                _infoLabel.text = @"";
                _boxView.frame = CGRectZero;
            });
            return ;
        }
        NSMutableArray *predictions = [NSMutableArray array];
        NSArray *orderedPredictions = [unorderedPredictions sortedArrayUsingComparator:^NSComparisonResult(id  _Nonnull obj1, id  _Nonnull obj2) {
            NSValue *predictionValue1 = obj1;
            NSValue *predictionValue2 = obj2;
            Prediction prediciton1;
            [predictionValue1 getValue:&prediciton1];
            Prediction prediciton2;
            [predictionValue2 getValue:&prediciton2];
            if (prediciton1.confidence > prediciton2.confidence) {
                return NSOrderedAscending;
            } else {
                return NSOrderedDescending;
            }
        }];
        NSMutableArray *keep = [NSMutableArray array];
        for (int i = 0; i < orderedPredictions.count; i++) {
            [keep addObject:[NSNumber numberWithBool:true]];
        }
        
        for (int i = 0; i < orderedPredictions.count; i++) {
            if (keep[i]) {
                [predictions addObject:orderedPredictions[i]];
                NSValue *predictionValue = orderedPredictions[i];
                Prediction prediction;
                [predictionValue getValue:&prediction];
                CGRect bbox1 = prediction.boundingBox;
                for (int j = i+1; j < orderedPredictions.count; j++) {
                    if (keep[j]) {
                        NSValue *predictionValueJ = orderedPredictions[i];
                        Prediction predictionJ;
                        [predictionValueJ getValue:&predictionJ];
                        CGRect bboxJ = predictionJ.boundingBox;
                        if ([self ioUwithA:bbox1 b:bboxJ] > nms_threshold) {
                            [keep replaceObjectAtIndex:j withObject:[NSNumber numberWithBool:false]];
                        }
                    }
                }
            }
        }
        
        dispatch_async(dispatch_get_main_queue(), ^{
            NSValue *dolaamenPredictionValue = [predictions firstObject];
            for (NSValue *predictionValue in predictions) {
                Prediction doraemonPrediciton;
                [predictionValue getValue:&doraemonPrediciton];
                CGRect rect = CGRectMake(doraemonPrediciton.boundingBox.origin.x * width, doraemonPrediciton.boundingBox.origin.y * height, CGRectGetWidth(doraemonPrediciton.boundingBox) * width, CGRectGetHeight(doraemonPrediciton.boundingBox) * height);
                NSLog(@"==> %f,%f,%f,%f",rect.origin.x,rect.origin.y,rect.size.width,rect.size.height);
            }
            Prediction dolaamenPrediction;
            [dolaamenPredictionValue getValue:&dolaamenPrediction];
            self.boxView.frame = CGRectMake(dolaamenPrediction.boundingBox.origin.x * width, dolaamenPrediction.boundingBox.origin.y * height  , CGRectGetWidth(dolaamenPrediction.boundingBox) * width, CGRectGetHeight(dolaamenPrediction.boundingBox) * height);
            
            if (dolaamenPrediction.labelIndex == 0) {
                _infoLabel.text = [NSString stringWithFormat:@"哆啦A梦置信度:%f",dolaamenPrediction.confidence];
            }
            //            } else {
            //                _infoLabel.text = [NSString stringWithFormat:@"%ld个pen置信度:%f",predictions.count,dolaamenPrediction.confidence];
            //            }
            
            if (detectionNum == 5) {
                detectionNum = 0;
                isDetection = NO;
                CGRect convertedRect = [captureVideoPreviewLayer metadataOutputRectOfInterestForRect:self.boxView.frame];
                convertedRect = CGRectMake(convertedRect.origin.x , 1 - convertedRect.origin.y, convertedRect.size.width, convertedRect.size.height);
                VNDetectedObjectObservation * newObservation = [VNDetectedObjectObservation observationWithBoundingBox:convertedRect];
                self.lasetObservation = newObservation;
            } else {
                detectionNum ++;
            }
            
            [self changeFocusWithPoint:CGPointZero];
            
        });
    }];
    request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
    VNImageRequestHandler *requestHandler = [[VNImageRequestHandler alloc]initWithCGImage:newPhoto.CGImage options:@{}];
    NSError *error = nil;
    [requestHandler performRequests:@[request] error:&error];
    if (error) {
        NSLog(@"%@",error.localizedDescription);
    }
}

- (float)ioUwithA:(CGRect)rectA b:(CGRect)rectB{
    float areaA = rectA.size.width * rectA.size.height;
    if (areaA < 0) {
        return 0;
    }
    float areaB = rectB.size.width * rectB.size.height;
    if (areaB < 0) {
        return 0;
    }
    
    float intersectionMinX = MAX(CGRectGetMinX(rectA), CGRectGetMinX(rectB));
    float intersectionMinY = MAX(CGRectGetMinY(rectA), CGRectGetMinY(rectB));
    float intersectionMaxX = MIN(CGRectGetMaxX(rectA), CGRectGetMaxX(rectB));
    float intersectionMaxY = MIN(CGRectGetMaxY(rectA), CGRectGetMaxY(rectB));
    float intersectionArea = MAX(intersectionMaxY - intersectionMinY, 0) *
    MAX(intersectionMaxX - intersectionMinX, 0);
    return  (intersectionArea / (areaA + areaB - intersectionArea));
}

#pragma mark - objectTracking
- (void)objectTrackingWithSamplebuffer:(CMSampleBufferRef)sampleBuffer{
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!self.lasetObservation) {
        return;
    }
    void(^ VNRequestCompletionHandler)(VNRequest *request, NSError * _Nullable error) = ^(VNRequest *request, NSError * _Nullable error){
        dispatch_async(dispatch_get_main_queue(), ^{
            VNDetectedObjectObservation *newObservation = request.results.firstObject;
            if (!newObservation) {
                _sequenceHandler = nil;
                isDetection = YES;
                AVCaptureDevice * camera = videoCaptureDeviceInput.device;
                if ([camera lockForConfiguration:nil]) {
                    [camera setExposurePointOfInterest:CGPointMake(0.5, 0.5)];
                    [camera setExposureMode:(AVCaptureExposureMode)AVCaptureExposureModeAutoExpose];
                    [camera unlockForConfiguration];
                }
                return ;
            }
            self.lasetObservation = newObservation;
            if (newObservation.confidence <= 0.3) {
                self.boxView.frame = CGRectZero;
                self.lasetObservation = nil;
                isDetection = YES;
                AVCaptureDevice * camera = videoCaptureDeviceInput.device;
                if ([camera lockForConfiguration:nil]) {
                    [camera setExposurePointOfInterest:CGPointMake(0.5, 0.5)];
                    [camera setExposureMode:(AVCaptureExposureMode)AVCaptureExposureModeAutoExpose];
                }
                return;
            }
            self.infoLabel.text = [NSString stringWithFormat:@"置信度:%f",newObservation.confidence];
            CGRect transformedRect = newObservation.boundingBox;
            transformedRect.origin.y =  1 - transformedRect.origin.y;
            CGRect convertedRect = [captureVideoPreviewLayer rectForMetadataOutputRectOfInterest:transformedRect];
            self.boxView.frame = convertedRect;
            [self changeFocusWithPoint:CGPointZero];
            
            
        });
    };
    VNTrackObjectRequest *request = [[VNTrackObjectRequest alloc] initWithDetectedObjectObservation:self.lasetObservation completionHandler:VNRequestCompletionHandler];
    request.trackingLevel = VNRequestTrackingLevelAccurate;
    
    if (!_sequenceHandler) {
        _sequenceHandler = [[VNSequenceRequestHandler alloc] init];
    }
    
    [_sequenceHandler performRequests:@[request] onCVPixelBuffer:pixelBuffer error:nil];
    
}

#pragma mark - 摄像头初始化
- (void)createCamera{
    captureSession_ = [[AVCaptureSession alloc] init];
    if ([captureSession_ canSetSessionPreset:AVCaptureSessionPreset640x480]) {
        [captureSession_ setSessionPreset:AVCaptureSessionPreset640x480];
    }
    
    if (@available(iOS 10.0, *)) {
        captureDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionBack];
    } else {
        NSArray *cameras = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
        for (AVCaptureDevice *camera in cameras) {
            if ([camera position] == AVCaptureDevicePositionBack) {
                captureDevice = camera;
            }
        }
    }
    
    AVCaptureDevice *audioCaptureDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeAudio];
    
    audioCaptureDeviceInput = [[AVCaptureDeviceInput alloc] initWithDevice:audioCaptureDevice error:nil];
    
    //视频输入
    videoCaptureDeviceInput = [AVCaptureDeviceInput deviceInputWithDevice:captureDevice error:nil];
    
    if ([captureSession_ canAddInput:videoCaptureDeviceInput]) {
        [captureSession_ addInput:videoCaptureDeviceInput];
    }
    if ([captureSession_ canAddInput:audioCaptureDeviceInput]) {
        [captureSession_ addInput:audioCaptureDeviceInput];
    }
    
    captureVideoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
    captureVideoDataOutput.videoSettings =  [NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA] forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    if ([captureSession_ canAddOutput:captureVideoDataOutput]) {
        [captureSession_ addOutput:captureVideoDataOutput];
    }
    
    myQueue = dispatch_queue_create("myQueue", NULL);
    [captureVideoDataOutput setSampleBufferDelegate:self queue:myQueue];
    //    captureVideoDataOutput.alwaysDiscardsLateVideoFrames = YES;
    
    //音频输出
    captureAudioDataOutput = [[AVCaptureAudioDataOutput alloc] init];
    [captureAudioDataOutput setSampleBufferDelegate:self queue:myQueue];
    if ([captureSession_ canAddOutput:captureAudioDataOutput]) {
        [captureSession_ addOutput:captureAudioDataOutput];
    }
    
    // 视频连接
    videoConnection = [captureVideoDataOutput connectionWithMediaType:AVMediaTypeVideo];
    //    videoConnection.videoOrientation = AVCaptureVideoOrientationPortrait;
    
    /// 音频连接
    audioConnection = [captureAudioDataOutput connectionWithMediaType:AVMediaTypeAudio];
    
    [captureDevice lockForConfiguration:nil];
    [captureDevice setFocusMode:AVCaptureFocusModeAutoFocus];
    [captureDevice setExposureMode:AVCaptureExposureModeContinuousAutoExposure];
    [captureDevice unlockForConfiguration];
    //初始化预览图层
    captureVideoPreviewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:captureSession_];
    [captureVideoPreviewLayer setVideoGravity:AVLayerVideoGravityResizeAspect];
    captureVideoPreviewLayer.frame = self.view.frame;
    [self.view.layer addSublayer:captureVideoPreviewLayer];
    [captureSession_ startRunning];
}


//属性改变操作
- (void)changeDeviceProperty:(PropertyChangeBlock ) propertyChange{
    
    AVCaptureDevice * captureDevice = [videoCaptureDeviceInput device];
    NSError * error;
    //注意改变设备属性前一定要首先调用lockForConfiguration:调用完之后使用unlockForConfiguration方法解锁
    if ([captureDevice lockForConfiguration:&error]) {
        
        propertyChange(captureDevice);
        [captureDevice unlockForConfiguration];
        
    } else {
        NSLog(@"设置设备属性过程发生错误，错误信息：%@", error.localizedDescription);
    }
}

- (void)changeFocusWithPoint:(CGPoint)point{
    //将UI坐标转化为摄像头坐标
    CGPoint cameraPoint;
    if (point.x == 0) {
       cameraPoint = [captureVideoPreviewLayer captureDevicePointOfInterestForPoint:point];
    } else {
        cameraPoint = [captureVideoPreviewLayer captureDevicePointOfInterestForPoint:_boxView.center];
    }
    
    [self focusWithMode:AVCaptureFocusModeAutoFocus exposureMode:AVCaptureExposureModeAutoExpose atPoint:cameraPoint];
    [self setExposurePoint:cameraPoint];
}
//设置聚焦点
- (void)focusWithMode:(AVCaptureFocusMode)focusMode exposureMode:(AVCaptureExposureMode)exposureMode atPoint:(CGPoint)point{
    [self changeDeviceProperty:^(AVCaptureDevice *captureDevice) {
        if ([captureDevice isFocusModeSupported:focusMode]) {
            [captureDevice setFocusMode:AVCaptureFocusModeAutoFocus];
        }
        
        if ([captureDevice isFocusPointOfInterestSupported]) {
            [captureDevice setFocusPointOfInterest:point];
        }
        
        if ([captureDevice isExposureModeSupported:exposureMode]) {
            [captureDevice setExposureMode:AVCaptureExposureModeAutoExpose];
        }
        
        if ([captureDevice isExposurePointOfInterestSupported]) {
            [captureDevice setExposurePointOfInterest:point];
        }
        
    }];
}

//设置曝光
-(void)setExposurePoint:(CGPoint)exposurePoint
{
    AVCaptureDevice * camera = videoCaptureDeviceInput.device;
    if ([camera isExposurePointOfInterestSupported] && [camera isExposureModeSupported:AVCaptureExposureModeAutoExpose]) {
        NSError *error;
        CGPoint adjustedPoint = exposurePoint;
        adjustedPoint = CGPointMake(adjustedPoint.y, 1.0f - adjustedPoint.x);
        if ([camera lockForConfiguration:&error]) {
            [camera setExposurePointOfInterest:adjustedPoint];
            [camera setExposureMode:(AVCaptureExposureMode)AVCaptureExposureModeAutoExpose];
            [camera unlockForConfiguration];
        } else {
            NSLog(@" setExposurePoint:  Error: device can not be lock to Configure");
        }
    } else {
        NSLog(@"- setFocusPoint: Error:  exposurePoint or exposureMode does not be supported in this device");
    }
}

#pragma mark - samplebuffer 转 cgimage
// 通过抽样缓存数据创建一个UIImage对象
- (UIImage *) imageFromSampleBuffer:(CMSampleBufferRef) sampleBuffer
{
    // 为媒体数据设置一个CMSampleBuffer的Core Video图像缓存对象
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    // 锁定pixel buffer的基地址
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    // 得到pixel buffer的基地址
    void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    
    // 得到pixel buffer的行字节数
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    
    // 得到pixel buffer的宽和高
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    if (width == 0 || height == 0) {
        return nil;
    }
    
    // 创建一个依赖于设备的RGB颜色空间
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    // 用抽样缓存的数据创建一个位图格式的图形上下文（graphics context）对象
    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                 bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGAffineTransform transform = CGAffineTransformIdentity;
    CGContextConcatCTM(context, transform);
    
    // 根据这个位图context中的像素数据创建一个Quartz image对象
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    
    // 裁剪 图片
    struct CGImage *cgImage = CGImageCreateWithImageInRect(quartzImage, CGRectMake(0, 0, width, height));
    
    // 解锁pixel buffer
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    // 释放context和颜色空间
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    // 用Quartz image创建一个UIImage对象image
    UIImage *image = [UIImage imageWithCGImage:cgImage];
    
    // 释放Quartz image对象
    CGImageRelease(cgImage);
    CGImageRelease(quartzImage);
    return (image);
    
}

#pragma mark - UI
- (UILabel *)infoLabel{
    if (!_infoLabel) {
        _infoLabel = [[UILabel alloc] initWithFrame:CGRectMake(40, SCREENHEIGHT - 40, SCREENWIDTH - 80, 30)];
        _infoLabel.textColor = [UIColor redColor];
        _infoLabel.textAlignment = NSTextAlignmentCenter;
    }
    return _infoLabel;
}

- (UIView *)boxView{
    if (!_boxView) {
        _boxView = [[UIView alloc] initWithFrame:CGRectZero];
        _boxView.layer.borderColor = [UIColor redColor].CGColor;
        _boxView.layer.borderWidth = 2;
        _boxView.backgroundColor = [UIColor clearColor];
    }
    return _boxView;
}

- (void)viewWillAppear:(BOOL)animated{
    [captureSession_ startRunning];
}
- (void)viewDidDisappear:(BOOL)animated
{
    [captureSession_ stopRunning];
}
@end
