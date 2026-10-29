# ChatGPT 工作模式在 WSL 中持续“重新连接”事故复盘

- 日期：2026-08-01
- 环境：Windows 11、ChatGPT Windows 客户端、WSL `Ubuntu-24.04`、Clash 本地代理 `127.0.0.1:7890`
- 主要项目：`/home/leonhard/sci001`
- ChatGPT 安装包：`OpenAI.Codex_26.727.6591.0_x64__2p2nqsd0c76g0`
- Codex app-server：`0.146.0-alpha.9.2`
- 状态：代理继承问题已修复；工作模式启动依赖已恢复，最终端到端对话仍应以一次实际 `turn/completed` 为最终验收

## 1. 结论摘要

这次故障的根本原因不是 WSL 文件位置、项目大小、插件数量、桌面宠物、`node_repl`，也不是 ChatGPT 普通聊天服务本身。

根本原因是：**ChatGPT 桌面客户端直接启动 WSL 内的 Codex app-server 时，通过 `WSLENV` 向该进程传入了空的 Windows 大写代理变量 `HTTP_PROXY`、`HTTPS_PROXY` 和 `NO_PROXY`。这使工作模式的 Codex 网络请求没有经过 Clash，而是尝试直接连接公网；当前网络环境下直连持续超时，因此界面反复显示“重新连接”。**

普通 WSL shell 和从 shell 启动的 Codex CLI 没有这个问题，因为它们从 WSL 自动代理配置中得到了正确的代理变量。因此早期的 `curl` 和 CLI 测试都成功，却不能代表 ChatGPT 工作模式所启动的 app-server。

最终修复是在 Windows 当前用户环境中显式设置：

```text
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
NO_PROXY=localhost,127.0.0.1,::1,10.*,172.16.*,...,172.31.*,192.168.*
```

彻底重启 ChatGPT 后，新 app-server 正确继承了大写和小写代理变量，真实 socket 从“直连公网并卡在 `SYN-SENT`”变成“连接 `127.0.0.1:7890` 且状态为 `ESTAB`”。模型列表、Apps 和连接器接口随后均返回 `200 OK`。

## 2. 用户可见症状

1. 普通“聊天”模式可以正常回复。
2. “工作”模式长时间显示“重新连接”，回复极慢或完全没有回复。
3. `sci001` 和新建的 `wsl-test` 都会复现，因此不是单个项目损坏。
4. 等待期间，桌面宠物和本地插件仍能响应。
5. WSL 内的文件可以被定位，Codex app-server 也确实运行在 WSL 中。

这些症状说明本地界面、WSL 文件访问和普通 ChatGPT 会话并未整体失效；故障集中在工作模式的 WSL Codex 网络请求链路。

## 3. 实际运行链路

工作模式的关键链路如下：

```text
ChatGPT Windows 客户端
  -> wsl.exe
  -> Ubuntu-24.04 中的 Codex app-server
  -> https://chatgpt.com/backend-api/codex/responses
  -> 模型流式响应
```

普通聊天与工作模式不是同一条本地执行链。普通聊天正常，只能证明 ChatGPT 客户端本身能够联网；它不能证明 WSL 内由桌面客户端直接启动的 Codex app-server 也正确使用了代理。

## 4. 根本原因的直接证据

### 4.1 普通 WSL shell 的代理环境正确

普通 shell 中同时存在正确的大写和小写变量：

```text
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
```

因此以下测试能够成功：

- WSL `curl` 访问 `chatgpt.com`；
- 长时间 HTTP 流测试；
- 从 WSL shell 启动的 Codex CLI 使用相同账号和模型完成请求。

### 4.2 故障时 app-server 的环境不同

ChatGPT 直接启动的 app-server 在故障时实际得到的是：

```text
HTTP_PROXY=
HTTPS_PROXY=
NO_PROXY=
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
```

也就是说，大写变量被显式传为空值。与此同时，`WSLENV` 中包含 `HTTP_PROXY`、`HTTPS_PROXY` 和 `NO_PROXY`，表明这些 Windows 变量会被转交到 WSL 子进程。

从结果上看，该版本组合下的 Codex HTTP 路径没有采用已有的小写代理值，而是走了直连。内部代理变量的精确优先级没有通过源码确认，但真实 socket 已直接证明最终路由结果。

### 4.3 故障时 app-server 在直连公网

故障中的 Codex 进程出现了类似以下连接：

```text
10.10.3.180:<临时端口> -> <公网地址>:443  SYN-SENT
```

它没有连接 `127.0.0.1:7890`。日志同步记录：

```text
Request failed ... /backend-api/codex/responses
stream disconnected - retrying sampling request (1/5 ... 5/5)
sampling_error=request timed out
```

模型列表、Apps MCP、远程插件目录和分析事件等请求也出现连接超时。这就是界面持续“重新连接”的直接来源。

### 4.4 直连失败、经 Clash 成功

同一台 Windows 机器上对 Codex 接口进行对照测试：

```text
直连：8 秒连接超时，HTTP 000
经 127.0.0.1:7890：约 2.14 秒返回 HTTP 401
```

`401` 在未附带认证信息的探测中是正常的服务端响应，证明网络和 TLS 链路已经建立；直连的 `000` 则说明连接根本没有建立。

### 4.5 修复后的环境和 socket

设置 Windows 当前用户代理变量并彻底重启 ChatGPT 后，新 app-server 得到：

```text
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
NO_PROXY=localhost,127.0.0.1,::1,...
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
```

真实连接变为：

```text
127.0.0.1:<临时端口> -> 127.0.0.1:7890  ESTAB
```

随后观察到：

- `GET /backend-api/codex/models`：`200 OK`；
- `POST /backend-api/ps/apps/batch`：`200 OK`；
- `GET /backend-api/connectors/directory/list`：`200 OK`；
- `node_repl`：初始化成功；
- `codex_apps`：初始化成功；
- 远程插件同步：多次完成。

这些结果完成了“环境变量 -> 实际 socket -> 服务端响应”的证据闭环。

## 5. 为什么普通聊天正常，而工作模式失败

普通聊天和工作模式使用不同的本地运行层：

- 普通聊天由 ChatGPT 客户端自身处理，Windows 系统代理已经指向 Clash，因此可以正常联网。
- 工作模式需要额外启动 WSL Codex app-server。该进程的环境来自 ChatGPT 桌面进程和 `WSLENV`，当 Windows 用户环境中没有显式的 `HTTP_PROXY/HTTPS_PROXY` 时，它收到的是空的大写变量。
- 因此同一个应用里会出现“聊天正常、工作模式重连”的表面矛盾。

## 6. 为什么排查过程中失败了很多次

### 6.1 被启动警告吸引：`respect_system_proxy` 与 `node_repl`

最初看到：

```text
Under-development features enabled: respect_system_proxy
MCP client for node_repl failed to start
```

这些警告很显眼，因此被优先怀疑。但后续证据表明：

- `respect_system_proxy` 的第一条只是“不稳定功能已启用”的提示，隐藏提示不会改变网络行为；
- Windows 命令行启动时的 `node_repl` 路径警告与 WSL 工作模式不是同一执行上下文；
- 修复后的 WSL 日志明确显示 `node_repl` 成功初始化。

失败原因：把“最显眼的警告”误当成“导致模型请求超时的错误”，没有先关联到具体失败请求。

### 6.2 过度关注 WSL 文件路径和项目位置

因为项目位于 WSL，最初怀疑 Windows 客户端无法正确访问或修改 WSL 文件。但进程和日志显示：

- app-server 确实运行在 `Ubuntu-24.04`；
- 工作目录确实是 `/home/leonhard/sci001`；
- 新建 `wsl-test` 仍复现。

失败原因：文件访问路径是工作模式的必要条件，但不是本次网络流中断的原因。

### 6.3 重置 ChatGPT 和新建线程没有解决

重置可以清理客户端缓存、会话状态和部分 UI 状态。早期日志中也确实出现过：

- `needs_resume`；
- `markedStreaming=true`；
- `Conversation state not found`；
- 对未知 conversation 的 turn 事件。

这些状态可能加重“重新连接”的显示，但重置后全新线程仍复现。

失败原因：根因位于 Windows 用户环境到 WSL app-server 的代理继承，应用重置不会自动创建缺失的用户环境变量，因此重置无法修复。

### 6.4 怀疑插件、Apps MCP 或桌面宠物

远程插件和 Apps MCP 的日志中存在大量超时，因此一度怀疑插件初始化阻塞了工作模式。用户指出等待期间桌面宠物和本地插件仍能正常工作，这是重要反证。

后续发现插件、Apps MCP、模型列表和模型请求是在同一错误网络路由上同时失败。它们是受害者，而不是共同根因。

失败原因：把多个同时失败的上层功能当成原因，没有先检查它们共享的底层网络路径。

### 6.5 WSL `curl` 和 Codex CLI 测试造成了假阴性

这是排查中最关键的方法错误。

WSL `curl`、长流测试和 Codex CLI 都成功，于是网络一度被错误排除。但这些测试是从普通 shell 启动的；该 shell 具有正确的大写代理变量。ChatGPT 工作模式的 app-server 是由桌面客户端直接启动，实际环境不同。

失败原因：测试了“同一台机器、同一个 WSL、同一个账号”，却没有测试“同一个父进程、同一组环境变量、同一条网络路由”。环境相似不等于执行上下文相同。

改进原则：以后必须比较目标进程的 `/proc/<pid>/environ` 和真实 socket，不能只测试旁路 shell。

### 6.6 反复切换 `respect_system_proxy`

将 `respect_system_proxy` 设为 `false` 或 `true` 都没有解决问题。

失败原因：该设置不会替 Windows 父进程补出缺失的 `HTTP_PROXY/HTTPS_PROXY` 值。只要 app-server 仍收到空的大写变量，它仍可能走直连。

该设置最终保留为：

```toml
[features]
respect_system_proxy = true
```

同时使用：

```toml
suppress_unstable_features_warning = true
```

后者只隐藏提示，不承担修复作用。

### 6.7 错误地把 WinHTTP 判为根因

Windows 直连失败、显式 Clash 代理成功，而当时 `netsh winhttp show proxy` 显示 Direct，因此一度推断工作模式依赖 WinHTTP，并将 WinHTTP 指向 Clash。

重启后故障仍存在，证明该推断不成立。随后 socket 证据显示，WSL Codex 进程仍在直接访问公网，并没有因为 WinHTTP 设置而改变路由。

失败原因：对照测试证明了“需要代理”，但没有证明“目标进程使用 WinHTTP”。从相关性跳到了具体实现结论。

该系统级改动已撤销，当前状态为：

```text
Current WinHTTP proxy settings: Direct access (no proxy server)
```

### 6.8 只重启 WSL app-server 仍可能无效

在设置 Windows 用户环境变量之前或之后，仅终止 app-server 并不一定足够。若 ChatGPT 桌面父进程仍持有旧环境，它重新创建的子进程仍可能继承旧值；有时桌面端也不会立即自动拉起新进程。

失败原因：环境变量在进程创建时继承，不能指望已运行的父进程动态更新其环境块。

正确做法：写入用户环境后，彻底退出并重新启动 ChatGPT，使新的父进程和 WSL 子进程共同继承新值。

## 7. 最终实施的改动

### 7.1 Windows 当前用户环境

位置：

```text
HKEY_CURRENT_USER\Environment
```

写入：

```text
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
NO_PROXY=localhost,127.0.0.1,::1,10.*,172.16.*,...,172.31.*,192.168.*
```

影响范围：以后由该 Windows 用户启动、且识别这些变量的程序会使用 Clash。Clash 必须监听 `127.0.0.1:7890`；如果 Clash 端口改变，需要同步修改变量。

### 7.2 Codex 配置

文件：

```text
C:\Users\26330\.codex\config.toml
```

关键项：

```toml
suppress_unstable_features_warning = true

[features]
respect_system_proxy = true

[desktop]
runCodexInWindowsSubsystemForLinux = true
```

WSL 工作模式已经确认启动于 `Ubuntu-24.04`，并可使用 `/home/leonhard/sci001`。

### 7.3 已撤销的改动

WinHTTP 已恢复为 Direct，避免给系统服务留下无必要的全局代理设置。

## 8. 当前仍可见、但不是本次根因的警告

日志中仍可能看到以下非阻断警告：

- 某些插件 manifest 的图标路径或默认 prompt 不符合当前限制；
- 已配置的某些运行时插件在刷新后的 marketplace 中未被发现；
- 推荐插件接口偶发请求失败；
- `devicecheck-attestation unavailable errorCode=1`；
- Windows 包中 `bwrap` 资源定位失败；
- `apps_mcp_path_override` 被识别为无效实验键。

判断它们不是本次根因的依据是：在这些警告仍存在时，模型列表、Apps、连接器和 MCP 初始化已经成功，并且 app-server 的网络 socket 已正确通过 Clash。

它们可以单独清理，但不应再与“工作模式持续重连”的主故障混为一谈。

## 9. 验收标准

修复不能只以“界面暂时不显示重连”为标准。完整验收应包括：

1. app-server 环境中大写 `HTTP_PROXY/HTTPS_PROXY` 均为 `http://127.0.0.1:7890`；
2. app-server 的外部连接实际指向 `127.0.0.1:7890`；
3. 模型列表接口返回 `200 OK`；
4. Apps 和连接器接口返回 `200 OK`；
5. `node_repl` 和 `codex_apps` 初始化成功；
6. 在 `/home/leonhard/sci001` 发送一个简单工作模式请求；
7. 日志中出现该 turn 的模型响应和 `turn/completed`，且没有 `request timed out` 或 1/5 至 5/5 的重试链。

本报告生成时，第 1 至第 5 项已经通过。第 6 至第 7 项应以用户随后的一次真实工作模式测试作为最终闭环。

## 10. 后续建议

### 10.1 Clash 端口变化时

若 Clash 不再使用 `7890`，同步更新 Windows 用户环境中的：

```text
HTTP_PROXY
HTTPS_PROXY
```

然后彻底重启 ChatGPT。

### 10.2 Clash 未运行时

由于代理变量是用户级持久设置，Clash 关闭后，识别这些变量的命令行工具和 ChatGPT 工作模式可能无法联网。此时应启动 Clash、修改端口，或移除这些变量。

### 10.3 回滚用户环境变量

如需完全撤销本次用户级代理设置，可在命令提示符中执行：

```bat
reg delete HKCU\Environment /v HTTP_PROXY /f
reg delete HKCU\Environment /v HTTPS_PROXY /f
reg delete HKCU\Environment /v NO_PROXY /f
```

执行后需要彻底退出并重新启动 ChatGPT。删除前应确认没有其他程序依赖这些用户变量。

### 10.4 再次出现重连时的最短诊断路径

1. 找到 WSL 中 Codex app-server 的 PID；
2. 查看 `/proc/<pid>/environ` 中代理值；
3. 用 `ss -tpn` 查看该 PID 的真实目标地址；
4. 若连接公网而不是 `127.0.0.1:7890`，先修复环境继承；
5. 若已经连接 Clash，再检查 `/backend-api/codex/responses` 的具体状态码和流中断原因；
6. 只有在网络路径已确认正确后，再排查插件、会话状态和 MCP。

## 11. 本次排查的主要教训

1. **不要把警告数量当成根因优先级。** 必须把错误与失败请求、进程和时间点关联起来。
2. **旁路测试成功不等于目标进程成功。** 同一个 WSL 中的 shell 和桌面客户端直接启动的进程可能拥有不同环境。
3. **代理问题必须看真实 socket。** 配置文件、环境变量和系统代理页面都只是意图，socket 才是实际行为。
4. **区分“证明需要代理”和“证明使用哪套代理机制”。** 本次 WinHTTP 误判正是混淆了这两件事。
5. **先证实共享底层，再处理上层功能。** 模型、插件、Apps MCP 同时失败时，应先查共同网络链路。
6. **环境变量修复需要重启父进程。** 只重启子进程不一定会得到新的 Windows 用户环境。

## 12. 责任说明

排查过程中，多次基于不完整证据提前宣布“已找到根因”，尤其是 WinHTTP 判断，给用户造成了重复重启和无效测试。这属于诊断方法上的问题：没有在下结论前完成“目标进程环境、真实 socket、请求结果”三项闭环。

最终根因是在比较普通 WSL shell 与 ChatGPT app-server 的 `/proc/<pid>/environ`，并检查该 PID 的实际 socket 后才被确认。今后同类问题应直接从目标进程开始，避免再次经历相同的试错链。

## 13. 2026-08-21 补充：切换到 Hiddify 后故障复发

> 本节记录 2026-08-21 从 Clash 切换到 Hiddify 后的第二次事故。前面第 1 至第 12 节中的 `127.0.0.1:7890` 是 Clash 时期的历史配置；当前 Hiddify 的本地 HTTP/混合代理端口为 `127.0.0.1:12334`。

### 13.1 结论摘要

这次故障仍然属于代理路由问题，但与 2026-08-01 的“Windows 父进程传入空的大写代理变量”不同。

切换到 Hiddify 后，Windows 用户环境和系统代理曾继续指向 Clash 的旧端口 `7890`。将 Windows 用户变量更新为 `12334` 后，Windows 侧已经正确，但 WSL shell 和 Codex app-server 仍反复得到 `7890`。即使重启 ChatGPT、终止 Ubuntu、完整执行 `wsl.exe --shutdown`、关闭 WSL `autoProxy` 并通过 `WSLENV` 显式传入 `12334`，登录 shell 仍显示旧端口。

最终根因是：

```text
/home/leonhard/.zshenv
```

中硬编码了 Clash 的旧代理：

```text
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
```

`.zshenv` 会在 zsh 启动时执行，并把 Windows/WSL 已经正确传入的 `12334` 再次覆盖为 `7890`。因此，之前所有只针对 Windows 环境、WSL 自动代理或进程重启的操作都无法真正消除旧值。

将 `.zshenv` 中两处端口改为 `12334` 后，Windows、WSL 原始环境、bash、zsh、Codex CLI 和桌面 app-server 的代理值终于统一，端到端网络恢复。

### 13.2 Hiddify 端口的确认

Windows 监听端口检查显示，同一个进程监听了：

```text
127.0.0.1:12334
127.0.0.1:12337
```

从 WSL 进行对照测试：

- 经 `127.0.0.1:12334` 访问 `https://chatgpt.com`：成功连接代理并收到远端 HTTP `403`；
- 经 `127.0.0.1:12337`：代理 CONNECT 超时；
- 经旧端口 `127.0.0.1:7890`：立即 `Connection refused`；
- WSL 直连公网：连接超时。

这里的 `403` 是远端服务已经响应的证据，说明 TCP、代理转发和 TLS 路径均已建立。由此确认 `12334` 是当前可用的 Hiddify HTTP/混合代理端口，`12337` 不能作为 `HTTP_PROXY/HTTPS_PROXY` 使用。

### 13.3 本次用户可见症状和日志证据

`caha` 项目中的真实请求出现：

```text
POST https://chatgpt.com/backend-api/codex/responses
Connection refused (os error 111)
stream disconnected - retrying sampling request
```

同一时间还观察到：

- 模型列表刷新失败；
- Apps MCP 初始化失败；
- 远程插件目录请求失败；
- app-server remote-control WebSocket 无法刷新或连接；
- 界面持续重新连接或请求无响应。

这些上层功能共同失败，是因为它们都被路由到已经没有进程监听的 `127.0.0.1:7890`，不是模型、`caha` 项目或插件本身损坏。

### 13.4 为什么前几次修复没有生效

#### 13.4.1 只修改 Windows 用户变量不够

Windows 用户级注册表已经变为：

```text
HTTP_PROXY=http://127.0.0.1:12334
HTTPS_PROXY=http://127.0.0.1:12334
```

但已运行的 Windows 父进程仍可能保留旧环境。诊断时曾观察到“用户级变量是 `12334`，当前 Codex 进程和 WSL app-server 仍是 `7890`”。这说明注册表值正确不能代替目标进程环境检查。

#### 13.4.2 重启 ChatGPT、终止发行版和关闭整个 WSL 都不够

依次尝试过：

```text
wsl.exe --terminate Ubuntu-24.04
wsl.exe --shutdown
```

重启后 WSL shell 仍恢复为 `7890`。这是因为旧值不是仅存在于 WSL 虚拟机缓存中，而是在每次 shell 启动时由 `.zshenv` 重新写入。

#### 13.4.3 `autoProxy` 不是最终根因

`.wslconfig` 原先为：

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
```

由于 Windows 系统代理已经是 `12334`、WSL 却仍显示 `7890`，一度怀疑 `autoProxy` 缓存旧值。将其设为 `false` 并完整关闭 WSL 后，登录 shell 仍然变回 `7890`，证明 `autoProxy` 不是最终覆盖源。

#### 13.4.4 显式配置 `WSLENV` 仍被 shell 覆盖

Windows 用户级 `WSLENV` 被扩展为：

```text
DEEPSEEK_API_KEY/u:HTTP_PROXY/u:HTTPS_PROXY/u:NO_PROXY/u
```

使用 `wsl.exe --exec /usr/bin/env` 直接查看 WSL 原始环境时，代理已经正确为 `12334`；但进入登录 shell 后又变为 `7890`。这个“原始环境正确、登录 shell 错误”的对照最终把排查范围缩小到 shell 启动文件。

随后搜索 zsh、bash、Conda 和 systemd 用户环境，发现只有 `.zshenv` 明确写有 `7890`。Conda 环境变量、Conda proxy 配置、PowerShell profile、Windows 系统级代理变量均不是本次覆盖源。

### 13.5 最终实施的改动

#### 13.5.1 Windows 用户代理

```text
HTTP_PROXY=http://127.0.0.1:12334
HTTPS_PROXY=http://127.0.0.1:12334
```

原有 `NO_PROXY` 保留。

#### 13.5.2 Windows 用户级 `WSLENV`

```text
DEEPSEEK_API_KEY/u:HTTP_PROXY/u:HTTPS_PROXY/u:NO_PROXY/u
```

这样由 Windows 直接启动的 WSL 进程可以显式获得大写代理变量。

#### 13.5.3 WSL 全局配置

文件：

```text
C:\Users\26330\.wslconfig
```

当前配置：

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=false
```

关闭 `autoProxy` 不是修复 `.zshenv` 覆盖的必要条件，但当前保留该设置，使代理来源由显式环境变量控制，避免自动代理与手动变量形成两个可能漂移的来源。

#### 13.5.4 zsh 启动环境

文件：

```text
/home/leonhard/.zshenv
```

将以下两项从 `7890` 改为 `12334`：

```text
http_proxy=http://127.0.0.1:12334
https_proxy=http://127.0.0.1:12334
```

大写 `HTTP_PROXY/HTTPS_PROXY` 继续由对应的小写变量派生，因此大小写值保持一致。

### 13.6 验收结果

修复后完成了以下验证：

1. Windows 用户环境中的 `HTTP_PROXY/HTTPS_PROXY` 均为 `12334`；
2. `wsl.exe --exec /usr/bin/env` 的原始 WSL 环境为 `12334`；
3. bash 和 zsh 中大小写代理变量均为 `12334`；
4. WSL 默认 `curl` 经 `12334` 收到远端 HTTP `403`，不再出现本地 `Connection refused`；
5. 在 `caha` 工作目录运行只读、临时 Codex CLI 测试，成功返回 `CODEX_PROXY_OK`；
6. Codex 桌面 app-server 的 `/proc/<pid>/environ` 中大小写代理变量均为 `12334`；
7. app-server 出现到 `127.0.0.1:12334` 的真实 `ESTAB` socket；
8. `GET /backend-api/codex/models` 返回 `200 OK`；
9. `POST /backend-api/ps/apps/batch` 返回 `200 OK`；
10. app-server remote-control WebSocket 状态变为 `Connected`；
11. Apps MCP 成功初始化。

启动期间仍出现个别插件 manifest 警告和一次远程插件目录刷新失败，但在模型接口、Apps 接口、WebSocket、真实 socket 和 Codex CLI 请求均成功的情况下，它们不属于本次主故障。

桌面端真实对话的最终界面验收，仍应在 `caha` 中发送一条消息，并确认日志出现模型响应和 `turn/completed`。网络层和 app-server 层已经完成闭环。

### 13.7 以后切换代理软件时的最短操作路径

1. 先确认新代理软件实际监听的 HTTP/混合端口，不使用软件名称推测端口；
2. 显式测试该端口，并区分“远端返回 401/403”和“本地 Connection refused/timeout”；
3. 搜索所有启动文件中的旧端口，至少包括 `.zshenv`、`.zshrc`、`.profile`、`.bashrc`、Conda activate 脚本和 systemd 用户环境；
4. 同步更新 Windows 用户环境、`WSLENV` 和仍保留的 shell 代理配置；
5. 完整关闭并重启 WSL/ChatGPT 后，分别检查原始 WSL 环境、登录 shell 环境和 app-server 环境；
6. 用 `ss -tpn` 确认 app-server 的真实连接目标；
7. 以一次真实 Codex 请求和 `turn/completed` 作为最终验收。

本次最重要的新教训是：**当 WSL 原始环境正确、登录 shell 环境错误时，应立即检查 shell 启动文件，不应继续反复修改 Windows 代理或重启 WSL。**

### 13.8 密钥安全补充

排查过程中发现 `.zshenv` 和 `.bashrc` 中存在明文 API 密钥。诊断命令一度读取并显示了这些行。报告不记录任何密钥值，但相关密钥应视为已经暴露并尽快撤销、轮换。

后续排查代理配置时，应只输出代理相关行，避免直接打印包含密钥的整个 shell 启动文件。长期建议将密钥迁移到权限受控的专用凭据文件或密钥管理机制中，而不是与普通 shell 配置混放。
