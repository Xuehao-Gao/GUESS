import torch
import torch.nn as nn
import torch.nn.functional as F

class humanml_init_s1_to_s2(nn.Module):

    def __init__(self):
        super().__init__()
        self.head = [15,12]
        self.upper_torso = [9,6]
        self.lower_torso = [3,0]
        self.left_arm_up = [13, 16]
        self.left_arm_down = [18, 20]
        self.right_arm_up = [14,17]
        self.right_arm_down = [19,21]
        self.left_leg_up = [1, 4]
        self.left_leg_down = [7, 10]
        self.right_leg_up = [2,5]
        self.right_leg_down = [8,11]


    def forward(self, s1_input):
        bs, n_frame, n_s1_joint, n_channel = s1_input.size()  # [64, 256, 7, 10]

        s2_head = F.avg_pool2d(s1_input[:, :, self.head, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_upper_torso = F.avg_pool2d(s1_input[:, :, self.upper_torso, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_lower_torso = F.avg_pool2d(s1_input[:, :, self.lower_torso, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_left_arm_up = F.avg_pool2d(s1_input[:, :, self.left_arm_up, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_left_arm_down = F.avg_pool2d(s1_input[:, :, self.left_arm_down, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_right_arm_up = F.avg_pool2d(s1_input[:, :, self.right_arm_up, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_right_arm_down = F.avg_pool2d(s1_input[:, :, self.right_arm_down, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_left_leg_up = F.avg_pool2d(s1_input[:, :, self.left_leg_up, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_left_leg_down = F.avg_pool2d(s1_input[:, :, self.left_leg_down, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_right_leg_up = F.avg_pool2d(s1_input[:, :, self.right_leg_up, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_right_leg_down = F.avg_pool2d(s1_input[:, :, self.right_leg_down, :], kernel_size=(2, 1))  # [N, C, T, V=1]

        s2_output = torch.cat((s2_head, s2_upper_torso, s2_lower_torso, s2_left_arm_up, s2_left_arm_down, s2_right_arm_up, s2_right_arm_down,
                            s2_left_leg_up, s2_left_leg_down, s2_right_leg_up, s2_right_leg_down), dim=-2)  # [N, C, T

        return s2_output.view(bs, n_frame, -1)


class humanml_init_s1_to_s3(nn.Module):

    def __init__(self):
        super().__init__()
        self.head = [15, 12]
        self.torso = [9, 6, 3, 0]
        self.left_arm = [13, 16, 18, 20]
        self.right_arm = [14, 17, 19, 21]
        self.left_leg = [1, 4, 7, 10]
        self.right_leg = [2, 5, 8, 11]

    def forward(self, s1_input):
        bs, n_frame, n_s1_joint, n_channel = s1_input.size()  # [64, 256, 7, 10]

        s3_head = F.avg_pool2d(s1_input[:, :, self.head, :], kernel_size=(2, 1))  # [N, T, V=1, 3]
        s3_torso = F.avg_pool2d(s1_input[:, :, self.torso, :], kernel_size=(4, 1))  # [N, T, V=1, 3]
        s3_left_arm = F.avg_pool2d(s1_input[:, :, self.left_arm, :], kernel_size=(4, 1))  # [N, T, V=1, 3]
        s3_right_arm = F.avg_pool2d(s1_input[:, :, self.right_arm, :], kernel_size=(4, 1))  # [N, T, V=1, 3]
        s3_left_leg = F.avg_pool2d(s1_input[:, :, self.left_leg, :], kernel_size=(4, 1))  # [N, T, V=1, 3]
        s3_right_leg = F.avg_pool2d(s1_input[:, :, self.right_leg, :], kernel_size=(4, 1))  # [N, T, V=1, 3]

        s3_output = torch.cat((s3_head, s3_torso, s3_left_arm, s3_right_arm, s3_left_leg, s3_right_leg), dim=-2)  # [N, T, V=6, 3]

        return s3_output.view(bs, n_frame, -1)


class humanml_init_s1_to_s4(nn.Module):

    def __init__(self):
        super().__init__()
        self.full_body_node = [0,1,2,3,4,5,6,7,8,9,10,11,
                               12,13,14,15,16,17,18,19,20,21]
    def forward(self, s1_input):
        bs, n_frame, n_s1_joint, n_channel = s1_input.size()  # [64, 256, 7, 10]

        s4_full_body_node = F.avg_pool2d(s1_input[:, :, self.full_body_node, :], kernel_size=(22, 1))  # [N, T, V=1, 3]


        return s4_full_body_node.view(bs, n_frame, -1)


class kit_init_s1_to_s2(nn.Module):

    def __init__(self):
        super().__init__()
        self.head = [4]
        self.upper_torso = [2,3]
        self.lower_torso = [1,0]
        self.left_arm_up = [5]
        self.left_arm_down = [6,7]
        self.right_arm_up = [8]
        self.right_arm_down = [9,10]
        self.left_leg_up = [11, 12]
        self.left_leg_down = [13, 14, 15]
        self.right_leg_up = [16,17]
        self.right_leg_down = [18,19,20]


    def forward(self, s1_input):
        bs, n_frame, n_s1_joint, n_channel = s1_input.size()  # [64, 256, 7, 10]

        s2_head = F.avg_pool2d(s1_input[:, :, self.head, :], kernel_size=(1, 1))  # [N, C, T, V=1]
        s2_upper_torso = F.avg_pool2d(s1_input[:, :, self.upper_torso, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_lower_torso = F.avg_pool2d(s1_input[:, :, self.lower_torso, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_left_arm_up = F.avg_pool2d(s1_input[:, :, self.left_arm_up, :], kernel_size=(1, 1))  # [N, C, T, V=1]
        s2_left_arm_down = F.avg_pool2d(s1_input[:, :, self.left_arm_down, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_right_arm_up = F.avg_pool2d(s1_input[:, :, self.right_arm_up, :], kernel_size=(1, 1))  # [N, C, T, V=1]
        s2_right_arm_down = F.avg_pool2d(s1_input[:, :, self.right_arm_down, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_left_leg_up = F.avg_pool2d(s1_input[:, :, self.left_leg_up, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_left_leg_down = F.avg_pool2d(s1_input[:, :, self.left_leg_down, :], kernel_size=(3, 1))  # [N, C, T, V=1]
        s2_right_leg_up = F.avg_pool2d(s1_input[:, :, self.right_leg_up, :], kernel_size=(2, 1))  # [N, C, T, V=1]
        s2_right_leg_down = F.avg_pool2d(s1_input[:, :, self.right_leg_down, :], kernel_size=(3, 1))  # [N, C, T, V=1]

        s2_output = torch.cat((s2_head, s2_upper_torso, s2_lower_torso, s2_left_arm_up, s2_left_arm_down, s2_right_arm_up, s2_right_arm_down,
                            s2_left_leg_up, s2_left_leg_down, s2_right_leg_up, s2_right_leg_down), dim=-2)  # [N, C, T

        return s2_output.view(bs, n_frame, -1)


class kit_init_s1_to_s3(nn.Module):

    def __init__(self):
        super().__init__()
        self.head = [4]
        self.torso = [0,1,2,3]
        self.left_arm = [5,6,7]
        self.right_arm = [8,9,10]
        self.left_leg = [11,12,13,14,15]
        self.right_leg = [16,17,18,19,20]

    def forward(self, s1_input):
        bs, n_frame, n_s1_joint, n_channel = s1_input.size()  # [64, 256, 7, 10]

        s3_head = F.avg_pool2d(s1_input[:, :, self.head, :], kernel_size=(1, 1))  # [N, T, V=1, 3]
        s3_torso = F.avg_pool2d(s1_input[:, :, self.torso, :], kernel_size=(4, 1))  # [N, T, V=1, 3]
        s3_left_arm = F.avg_pool2d(s1_input[:, :, self.left_arm, :], kernel_size=(3, 1))  # [N, T, V=1, 3]
        s3_right_arm = F.avg_pool2d(s1_input[:, :, self.right_arm, :], kernel_size=(3, 1))  # [N, T, V=1, 3]
        s3_left_leg = F.avg_pool2d(s1_input[:, :, self.left_leg, :], kernel_size=(5, 1))  # [N, T, V=1, 3]
        s3_right_leg = F.avg_pool2d(s1_input[:, :, self.right_leg, :], kernel_size=(5, 1))  # [N, T, V=1, 3]

        s3_output = torch.cat((s3_head, s3_torso, s3_left_arm, s3_right_arm, s3_left_leg, s3_right_leg), dim=-2)  # [N, T, V=6, 3]

        return s3_output.view(bs, n_frame, -1)



class kit_init_s1_to_s4(nn.Module):

    def __init__(self):
        super().__init__()

        self.full_body_node = [0,1,2,3,4,5,6,7,8,9,10,11,
                               12,13,14,15,16,17,18,19,20]
    def forward(self, s1_input):
        bs, n_frame, n_s1_joint, n_channel = s1_input.size()  # [64, 256, 7, 10]


        s4_full_body_node = F.avg_pool2d(s1_input[:, :, self.full_body_node, :], kernel_size=(21, 1))  # [N, T, V=1, 3]


        return s4_full_body_node.view(bs, n_frame, -1)


if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')
    bs, n_frame, n_s1_joint, n_channel = 6, 193, 22, 3
    model = init_s3_to_s1()
    x = torch.randn(bs, n_frame, n_s1_joint, n_channel)
    y = model.forward(x)

