# GitHub 私有仓库上传说明

本目录是 **/home/dgx/eno-fr3-github-stage**，不是原始 `/home/dgx/eno-isaac`。历史源码放在 `archive/legacy_workspace`，单孔核心代码在 `project/`，演示视频预留在 `media/demos`。

## 上传前检查

```bash
cd /home/dgx/eno-fr3-github-stage
find . -type f -size +20M -print
find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.zip' \) -print
find . -type l -print
du -sh .
```

自动扫描无法覆盖全部凭据、隐私、版权风险。应人工检查密钥、个人路径、第三方模型/资源许可。**不应上传导师/学长资料、内部服务器访问凭据或未经授权的数据。**

## 创建私有仓库

```bash
cd /home/dgx/eno-fr3-github-stage
git init -b main
git add .
git diff --cached --stat
git diff --cached --name-only
# 检查内容无误后：
git commit -m "Organize FR3 single-hole project and research archive"
# 请先确认 GitHub CLI 已登录：
gh auth status
gh repo create eno-fr3-peg-in-hole --private --source=. --remote=origin --push
```

若 `git` 身份未设置，按实际 GitHub 身份配置 `user.name` 与 `user.email`。若 `gh` 未安装，可在 GitHub 网页创建空的**私有**仓库后添加 remote 推送。不要在聊天中粘贴 token。

## 视频

未来视频放进 `media/demos/`，当前 `.gitignore` 会允许该目录下的 MP4。对大视频优先压缩，单个文件超过 GitHub 普通 Git 限制则使用 Git LFS、Release 或外部存储；上传成功后再在 README 增加实际链接。

## 分享给老师之前

建议的阅读顺序：`README.md` → `docs/PROJECT_MAP.md` → `docs/RESULTS.md` → 相关演示视频。

当前为**单孔闭环原型的工程展示**；六孔未完成、学习策略与物理成功验证仍有局限，不应以代码上传代替任务最终验收。
