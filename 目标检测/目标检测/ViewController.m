//
//  ViewController.m
//  目标检测
//
//  Created by 严玉鑫 on 2018/4/13.
//  Copyright © 2018年 严玉鑫. All rights reserved.
//

#import "ViewController.h"
#import "ObjectTrackDetectionViewController.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
}

- (IBAction)objectDetection:(id)sender {
    ObjectTrackDetectionViewController *objectVC = [[ObjectTrackDetectionViewController alloc] init];
    [self.navigationController pushViewController:objectVC animated:YES];
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
