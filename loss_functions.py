from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import torch

scale = 1.0

def multiScaleLoss(pred_flows, gt_flow, fps_idxs, alpha = [0.02, 0.04, 0.08, 0.16]):

    #num of scale
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1

    #generate GT list and mask1s
    gt_flows = [gt_flow]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
        gt_flows.append(sub_gt_flow)

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
        total_loss += alpha[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()

    return total_loss
    
def loss_fn_kd_2(outputs, fps_idxs, gt_flow, teacher_outputs, teacher_fps_idxs, gamma, alpha=[0.02, 0.04, 0.08, 0.16]):
    # student - teacher
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs)
    KD_loss += gamma*loss1 + (1-gamma)*loss2
    
    return KD_loss

def attentiveImitationLoss(outputs, fps_idxs, gt_flow, teacher_outputs, teacher_fps_idxs, t_history, gamma, alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss =torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss_ST = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs)
    loss_SG = multiScaleLoss(outputs, gt_flow, fps_idxs)
    loss_TG = multiScaleLoss(teacher_outputs, gt_flow, teacher_fps_idxs)


    sigma = 1 - ((loss_TG)/(max(t_history) - min(t_history)))

    KD_loss += gamma*(loss_SG) + (1-gamma)*sigma*(loss_ST)

    return KD_loss

def biDirectionLoss(outputs, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, teacher_fps_idxs, gamma1, gamma2, beta, alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)

    g_loss1 = multiScaleLoss(outputs, gt_flow, fps_idxs1)
    g_loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs2)

    k_loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    k_loss2 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs2)


    KD_loss += beta*(gamma1*k_loss1 + (1-gamma1)*g_loss1) + (1-beta)*(gamma2*k_loss2 + (1-gamma2)*g_loss2)
    return KD_loss


def loss_fn_ht(outputs, feat1s, fps_idxs1, fps_idxs2, gt_flow, teacher_outputs, t_feat1s, teacher_fps_idxs, gamma, layer=0,  alpha=[0.02, 0.04, 0.08, 0.16]):
    KD_loss = torch.zeros(1).cuda()

    teacher_outputs_0 = teacher_outputs[0].permute(0, 2, 1)
    loss1 = multiScaleLoss(outputs, teacher_outputs_0, fps_idxs1)
    loss2 = multiScaleLoss(outputs, gt_flow, fps_idxs1)

    Hint_loss = ((feat1s[3]-t_feat1s[3])**2)/2


    KD_loss += gamma*loss1 + (1-gamma)*loss2  + Hint_loss.sum()/(feat1s[0].nelement())

    return KD_loss
