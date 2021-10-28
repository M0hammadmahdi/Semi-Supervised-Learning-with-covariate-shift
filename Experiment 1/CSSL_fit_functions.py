import torch
import torch.nn.functional as F


# This functions contains fit function for training of different models
def fit_joint(train_dl, train_nolabel_dl, model, opt1, opt2, loss1, loss2, epochs):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss1(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt1.step()
            opt1.zero_grad()
        for xb, yb in train_nolabel_dl:
            pred = model(xb)
            loss = loss2(pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt2.step()
            opt2.zero_grad()


def fit_L(train_dl, model, opt, loss_func, epochs):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt.step()
            opt.zero_grad()


def fit_UL(train_dl, model, opt, loss_func, epochs):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred)
            # print(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt.step()
            opt.zero_grad()


def fit_EM(x_train_label, y_train_label, x_train_nolabel, model, opt, loss1, loss2, Beta, epochs, bs1, bs2):
    n = len(x_train_label)
    for epoch in range(epochs):
        for i in range((n - 1) // bs1 + 1):
            # print('Epoch: {}, Batch:{}'.format(epoch, i))
            start_i = i * bs1
            end_i = start_i + bs1
            xbL = x_train_label[start_i:end_i]
            yb = y_train_label[start_i:end_i]
            pred = model(xbL)
            pred1 = F.softmax(pred/100, 1)

            start_i = i * bs2
            end_i = start_i + bs2
            xbU = x_train_nolabel[start_i:end_i]
            predU = model(xbU)
            pred2 = F.softmax(predU/100, 1)

            loss = loss1(pred, yb) + Beta * (loss2(pred1) + loss2(pred2))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            opt.step()
            opt.zero_grad()


def fit_Upper(train_dl1, train_dl2, model, opt, loss_func, epochs):
    for epoch in range(epochs):
        for xb, yb in train_dl1:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt.step()
            opt.zero_grad()
        for xb, yb in train_dl2:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt.step()
            opt.zero_grad()


def fit_EM_1(x_train_label, y_train_label, x_train_nolabel, model, opt, loss1, loss2, Beta, epochs, bs1, bs2):
    n = len(x_train_label)
    for epoch in range(epochs):
        for i in range((n - 1) // bs1 + 1):
            # print('Epoch: {}, Batch:{}'.format(epoch, i))
            start_i = i * bs1
            end_i = start_i + bs1
            xbL = x_train_label[start_i:end_i]
            yb = y_train_label[start_i:end_i]
            pred = model(xbL)

            start_i = i * bs2
            end_i = start_i + bs2
            xbU = x_train_nolabel[start_i:end_i]
            predU = model(xbU)
            pred2 = F.softmax(predU, 1)

            loss = loss1(pred, yb) + Beta * loss2(pred2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            opt.step()
            opt.zero_grad()


def fit_Mahdi(x_train_label, y_train_label, x_train_nolabel, M1, M2, M3, opt, loss1, loss2, Beta, epochs, bs1, bs2):
    n = len(x_train_label)
    for epoch in range(epochs):
        for i in range((n - 1) // bs1 + 1):
            start_i = i * bs1
            end_i = start_i + bs1
            xb = x_train_label[start_i:end_i]
            yb = y_train_label[start_i:end_i]
            start_i = i * bs2
            end_i = start_i + bs2
            xU = x_train_nolabel[start_i:end_i]

            pred = M3(M1(xb))
            preds = F.softmax(pred, 1)
            pred1 = M3(M2(xU))
            pred2 = M3(M2(xb))
            pred1 = F.softmax(pred1, 1)
            loss = (1 - Beta) * (loss1(pred, yb) + loss1(pred2, yb)) + Beta * (loss2(torch.cat([pred1, preds], dim=0)))
            # print('Epoch:{}, Batch:{}, Loss:{}'.format(epoch, i, loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(M1.parameters(), 100)
            torch.nn.utils.clip_grad_norm_(M2.parameters(), 100)
            torch.nn.utils.clip_grad_norm_(M3.parameters(), 100)
            opt.step()
            opt.zero_grad()

"""
def fit(x_train_label, y_train_label, x_train_nolabel, model1, model2, opt1, opt2, loss1, loss2, Beta, c, epochs, bs1,
        bs2):
    n = len(x_train_label)
    end_2 = 0
    for epoch in range(epochs):
        for i in range((n - 1) // bs1 + 1):
            # print('Epoch: {}, Batch:{}'.format(epoch, i))
            start_i = i * bs1
            end_i = start_i + bs1
            xb = x_train_label[start_i:end_i]
            yb = y_train_label[start_i:end_i]

            st_1 = end_2
            end_1 = st_1 + bs2
            xU = x_train_nolabel[st_1:end_1]
            # yb = y_train_label[st_1:end_1]
            pred = model1(xb)
            preds = F.softmax(pred/1, 1)
            pred1 = model1(xU)
            pred1 = F.softmax(pred1/1, 1)
            loss = loss1(pred, yb) + Beta * (loss2(preds) + loss2(pred1))
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model1.parameters(), 100)
            opt1.step()
            opt1.zero_grad()

            model2.M_out.weight = model1.M_out.weight
            model2.M_out.bias = model1.M_out.bias

            xb = x_train_label[end_i:end_i + bs1]
            yb = y_train_label[end_i:end_i + bs1]
            pred = model2(xb)

            st_2 = end_1
            end_2 = st_2 + bs2 * c
            xU = x_train_nolabel[st_2:end_2]
            # yb = y_train_label[st_2:end_2]
            pred1 = model2(xU)
            pred1 = F.softmax(pred1, 1)

            loss = loss1(pred, yb) + Beta * c * (loss2(preds) + loss2(pred1))
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 100)
            opt2.step()
            opt2.zero_grad()

            model1.M_out.weight = model2.M_out.weight
            model1.M_out.bias = model2.M_out.bias
            # model1.M_out2.weight = model2.M_out2.weight
            # model1.M_out2.bias = model2.M_out2.bias
"""
