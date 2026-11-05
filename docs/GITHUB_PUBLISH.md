# 私有 GitHub 发布前检查

## 1. 先确认这是独立整理目录

推荐仓库目录：`/home/dgx/eno-fr3-github-stage`。不要在原 `eno-isaac` 的几百份实验和大文件上直接执行 `git add .`。

## 2. 校验准备提交的范围

```bash
cd /home/dgx/eno-fr3-github-stage
find . -type f | wc -l
du -sh .
# 定位所有超过 20 MB 的文件（正常的源码归档中不应有）
find . -type f -size +20M -print
# 查看符号链接（可能包含意料外的源文件路径）
find . -type l -print
# 检查二进制权重、压缩包、录像是否意外进入仓库
find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.zip' -o -name '*.mp4' \) -print
```

再人工检查源码里是否包含账户、凭据、私有 IP、绝对主机路径、联系人信息及没有发布权限的第三方资产。`REPO_SECRET_ALERTS=[]` 仅代表有限正则没有命中，**不是安全审计通过证明**。保留 SHA256 清单并不能代替提供必要运行资产。

## 3. 创建首个 Git 提交

```bash
cd /home/dgx/eno-fr3-github-stage
git init -b main
git status --short
git add .
git diff --cached --stat
# 最后再看一次将被提交的文件清单
git diff --cached --name-only | less
# 检查无误后才提交
git commit -m "Archive FR3 peg-in-hole research progress"
```

如果提示 `Author identity unknown`，只在确认身份后执行 `git config user.name` 与 `git config user.email` 设置，**不要照抄别人的身份**。

## 4. 创建私有 GitHub 仓库并推送

首先检查 `gh` 是否可用：

```bash
command -v gh
gh auth status
```

如已安装但未登录，执行 `gh auth login` 完成浏览器/设备授权；不要在聊天中粘贴 token。之后：

```bash
cd /home/dgx/eno-fr3-github-stage
gh repo create eno-fr3-peg-in-hole --private --source=. --remote=origin --push
```

如果 `gh` 未安装，也可以先在 GitHub 网页创建**空的私有仓库**，再按网页提供的地址执行 `git remote add origin ...` 和 `git push -u origin main`。不要把这里的仓库名当作已经创建成功。

## 5. 老师查看什么？

发私有仓库访问权限前，建议确认老师是否愿意使用 GitHub，并邀请对应 GitHub 账户为 Collaborator；不要直接公开仓库。展示顺序建议：`README.md` → `docs/PROJECT_MAP.md` → `docs/EXPERIMENTS.md` → `docs/PROGRESS.md`。如果补充演示视频，清楚注明该视频使用何种感知与控制版本以及物理成功判定。
