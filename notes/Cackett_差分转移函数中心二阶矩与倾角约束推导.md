# Cackett 两态差分转移函数：中心二阶矩与倾角约束推导

> 日期：2026-08-14  
> 更新：2026-08-17  
> 目的：从 Cackett et al. (2007) 第 3 节的两态差分转移函数出发，计算 $M_0$、$M_1$、$M_2$ 和归一化中心二阶矩，给出 $b=3/4$ 的准确数值系数，并说明如何用于倾角约束及 W2 参考波段下的 MICA 相对核。  
> 适用模型：平坦、轴对称、方位响应均匀、中心源高度忽略、单色响应，且所选波长远离盘内外边界。

---

## 0. 最终结果

定义

$$
r\equiv \frac{T_F}{T_B},
\qquad
\epsilon\equiv r^{2/b},
$$

$$
I_n(b)
\equiv
\int_0^\infty
\frac{x^{n/b-1}}{e^x-1}\,dx
=
\Gamma\!\left(\frac{n}{b}\right)
\zeta\!\left(\frac{n}{b}\right),
$$

以及

$$
\tau_\lambda^*
\equiv
\frac{R_0}{c}
\left(
\frac{\lambda kT_B}{hc}
\right)^{1/b}.
$$

则差分盘核的质心为

$$
\boxed{
\bar\tau_\lambda
=
\tau_\lambda^*
\frac{I_3}{I_2}
\frac{1-\epsilon^{3/2}}{1-\epsilon}
},
$$

原点二阶矩为

$$
\boxed{
\left\langle\tau^2\right\rangle_\lambda
=
(\tau_\lambda^*)^2
\left(1+\frac12\sin^2 i\right)
\frac{I_4}{I_2}(1+\epsilon)
},
$$

所以中心二阶矩是

$$
\boxed{
\sigma_{\tau,\lambda}^2
=
(\tau_\lambda^*)^2
\left[
\left(1+\frac12\sin^2 i\right)
\frac{I_4}{I_2}(1+\epsilon)
-
\left(
\frac{I_3}{I_2}
\frac{1-\epsilon^{3/2}}{1-\epsilon}
\right)^2
\right].
}
$$

倾角并不乘在整个方差上。它只增加

$$
\boxed{
\Delta_i\sigma_\tau^2
=
\frac{\sin^2i}{2}
\frac{\langle R^2\rangle_W}{c^2}
}
$$

这一项；质心本身与倾角无关。

---

## 1. 两态差分转移函数

亮态和暗态的温度剖面为

$$
T_B(R)
=
T_B\left(\frac{R}{R_0}\right)^{-b},
\qquad
T_F(R)
=
T_F\left(\frac{R}{R_0}\right)^{-b}.
$$

定义差分黑体权重

$$
\Delta B_\nu(R)
\equiv
B_\nu[T_B(R)]-B_\nu[T_F(R)].
$$

Cackett 采用的平坦盘时延为

$$
\tau(R,\phi)
=
\frac{R}{c}
\left(1+\sin i\cos\phi\right).
$$

于是两态差分转移函数为

$$
\Psi_\nu(\tau,\lambda)
=
\frac{\cos i}{D^2}
\int_0^\infty R\,dR
\int_0^{2\pi}d\phi\,
\Delta B_\nu(R)
\delta\!\left[
\tau-\frac{R}{c}
\left(1+\sin i\cos\phi\right)
\right].
$$

其中

$$
B_\nu(T)
=
\frac{2hc}{\lambda^3}
\frac{1}{\exp[hc/(\lambda kT)]-1}.
$$

定义未归一化时延矩

$$
M_n(\lambda)
\equiv
\int_{-\infty}^{+\infty}
\tau^n\Psi_\nu(\tau,\lambda)\,d\tau.
$$

利用 $\delta$ 函数，

$$
M_n
=
\frac{\cos i}{D^2c^n}
\int_0^\infty
R^{n+1}\Delta B_\nu(R)\,dR
\int_0^{2\pi}
\left(1+\sin i\cos\phi\right)^n d\phi.
$$

问题因此分成方位角几何积分和径向差分黑体积分。

为避免后文的“二阶矩”混淆，固定以下记号：

$$
M_n
\equiv
\int \tau^n\Psi(\tau)\,d\tau
$$

表示未归一化原点矩，

$$
m_n
\equiv
\frac{M_n}{M_0}
$$

表示归一化原点矩，并定义

$$
\mu\equiv m_1,
\qquad
V\equiv m_2-m_1^2.
$$

因此 $V$ 才是归一化中心二阶矩，也就是方差。当前项目代码中的 `moment2` 指的是 $V$，不是 $M_2$ 或 $m_2$。

---

## 2. 方位角积分：准确的倾角系数

定义

$$
A_n(i)
\equiv
\int_0^{2\pi}
\left(1+\sin i\cos\phi\right)^n d\phi.
$$

同时定义归一化方位因子

$$
\mathcal A_n(i)
\equiv
\frac{A_n(i)}{2\pi}.
$$

将二项式展开并利用奇数次 $\cos\phi$ 的方位平均为零，可得一般通式

$$
\boxed{
\mathcal A_n(i)
=
\sum_{j=0}^{\lfloor n/2\rfloor}
\binom{n}{2j}
\binom{2j}{j}
\frac{\sin^{2j}i}{4^j}
}.
$$

等价地，

$$
\boxed{
\mathcal A_n(i)
=
\sum_{j=0}^{\lfloor n/2\rfloor}
\frac{n!}{(n-2j)!(j!)^2\,4^j}
\sin^{2j}i
}.
$$

对于 $0\leq i<\pi/2$，还可以写成 Legendre 多项式或超几何函数：

$$
\boxed{
\mathcal A_n(i)
=
\cos^n i\,P_n(\sec i)
=
{}_2F_1\!\left(-\frac n2,\frac{1-n}{2};1;\sin^2i\right).
}
$$

若需要连续计算很多高阶矩，可以使用递推式

$$
\boxed{
(n+1)\mathcal A_{n+1}
=
(2n+1)\mathcal A_n
-n\cos^2i\,\mathcal A_{n-1},
}
$$

其中 $\mathcal A_0=\mathcal A_1=1$。

零阶为

$$
A_0=2\pi.
$$

一阶为

$$
\begin{aligned}
A_1
&=
\int_0^{2\pi}
\left(1+\sin i\cos\phi\right)d\phi\\
&=2\pi,
\end{aligned}
$$

因为 $\int_0^{2\pi}\cos\phi\,d\phi=0$。因此质心不依赖倾角。

二阶为

$$
\begin{aligned}
A_2
&=
\int_0^{2\pi}
\left(
1+2\sin i\cos\phi+\sin^2i\cos^2\phi
\right)d\phi\\
&=
2\pi+\pi\sin^2i\\
&=
2\pi\left(1+\frac12\sin^2i\right).
\end{aligned}
$$

所以

$$
\boxed{
\frac{A_2}{A_0}
=
1+\frac12\sin^2i
}.
$$

这个 $1/2$ 是倾角约束中最关键的几何系数。

再往后的几个系数为

$$
\mathcal A_3
=
1+\frac32\sin^2i,
$$

$$
\mathcal A_4
=
1+3\sin^2i+\frac38\sin^4i.
$$

---

## 3. 径向积分

定义

$$
J_n(\lambda)
\equiv
\int_0^\infty
R^{n+1}\Delta B_\nu(R)\,dR.
$$

这里的 $R^{n+1}$ 来自面积元中的 $R$ 和时延幂次中的 $R^n$。先对亮态或暗态中的任意一个状态 $S\in\{B,F\}$ 计算

$$
J_{n,S}
\equiv
\int_0^\infty
R^{n+1}B_\nu[T_S(R)]\,dR,
$$

其中

$$
T_S(R)
=
T_S\left(\frac{R}{R_0}\right)^{-b}.
$$

代入 Planck 函数后，

$$
J_{n,S}
=
\frac{2hc}{\lambda^3}
\int_0^\infty
\frac{R^{n+1}\,dR}
{\exp\!\left[
\dfrac{hc}{\lambda kT_S}
\left(\dfrac{R}{R_0}\right)^b
\right]-1}.
$$

首先令

$$
y=\frac{R}{R_0},
\qquad
q_S=\frac{\lambda kT_S}{hc}.
$$

因为

$$
R^{n+1}dR
=
R_0^{n+2}y^{n+1}dy,
$$

所以

$$
J_{n,S}
=
\frac{2hc}{\lambda^3}R_0^{n+2}
\int_0^\infty
\frac{y^{n+1}}{e^{y^b/q_S}-1}\,dy.
$$

再令

$$
x=\frac{y^b}{q_S}.
$$

于是

$$
y=(q_Sx)^{1/b},
$$

$$
dy
=
\frac1b q_S^{1/b}x^{1/b-1}dx,
$$

$$
y^{n+1}dy
=
\frac1b
q_S^{(n+2)/b}
x^{(n+2)/b-1}dx.
$$

因此

$$
\begin{aligned}
J_{n,S}
&=
\frac{2hc}{\lambda^3}
\frac{R_0^{n+2}}{b}
\left(\frac{\lambda kT_S}{hc}\right)^{(n+2)/b}\\
&\quad\times
\int_0^\infty
\frac{x^{(n+2)/b-1}}{e^x-1}\,dx.
\end{aligned}
$$

使用

$$
\int_0^\infty
\frac{x^{s-1}}{e^x-1}\,dx
=
\Gamma(s)\zeta(s),
\qquad
\operatorname{Re}(s)>1,
$$

并取 $s=(n+2)/b$，得到

$$
\boxed{
J_{n,S}
=
\frac{2hc}{\lambda^3}
\frac{R_0^{n+2}}{b}
\left(\frac{\lambda kT_S}{hc}\right)^{(n+2)/b}
I_{n+2}(b).
}
$$

亮态减暗态后，

$$
\boxed{
J_n
=
\frac{2hc}{\lambda^3}
\frac{R_0^{n+2}}{b}
\left(\frac{\lambda kT_B}{hc}\right)^{(n+2)/b}
I_{n+2}(b)
\left[
1-r^{(n+2)/b}
\right].
}
$$

由于

$$
r^{(n+2)/b}
=
\epsilon^{(n+2)/2},
$$

也可以写成

$$
\boxed{
J_n
=
\frac{2hc}{\lambda^3}
\frac{R_0^{n+2}}{b}
\left(\frac{\lambda kT_B}{hc}\right)^{(n+2)/b}
I_{n+2}(b)
\left[
1-\epsilon^{(n+2)/2}
\right].
}
$$

因此径向积分的关键换元链是

$$
R
\longrightarrow
y=\frac{R}{R_0}
\longrightarrow
x=
\frac{hc}{\lambda kT_S}
\left(\frac{R}{R_0}\right)^b,
$$

它把黑体径向积分化为标准的 $\Gamma\zeta$ 积分。

---

## 4. 一般原点矩与波长标度

把方位角和径向积分合并。

### 4.1 一般未归一化原点矩

利用 $A_n=2\pi\mathcal A_n$，一般的未归一化原点矩为

$$
\boxed{
\begin{aligned}
M_n
&=
\frac{4\pi hc}{bD^2\lambda^3}
\cos i\,
\frac{R_0^{n+2}}{c^n}
\left(\frac{\lambda kT_B}{hc}\right)^{(n+2)/b}\\
&\quad\times
I_{n+2}(b)
\left[1-\epsilon^{(n+2)/2}\right]
\mathcal A_n(i).
\end{aligned}
}
$$

在 $T_B/T_F$、$b$ 和 $i$ 不随波长改变时，$B_\nu$ 表示下有

$$
\boxed{
M_n(\lambda)
\propto
\frac{\cos i}{D^2}
\lambda^{(n+2)/b-3}.
}
$$

这里的 $-3$ 来自 $B_\nu=2hc\lambda^{-3}/(e^x-1)$。若改用 $B_\lambda$，未归一化矩的指数相应变为 $(n+2)/b-5$；归一化时延矩不受这一谱密度约定影响。

### 4.2 零阶矩

$$
\boxed{
M_0
=
\frac{4\pi hc}{\lambda^3}
\frac{R_0^2\cos i}{bD^2}
\left(\frac{\lambda kT_B}{hc}\right)^{2/b}
I_2(b)(1-\epsilon).
}
$$

### 4.3 一阶矩

$$
\boxed{
M_1
=
\frac{4\pi hc}{\lambda^3}
\frac{R_0^3\cos i}{bD^2c}
\left(\frac{\lambda kT_B}{hc}\right)^{3/b}
I_3(b)
\left(1-\epsilon^{3/2}\right).
}
$$

### 4.4 原点二阶矩

$$
\boxed{
M_2
=
\frac{4\pi hc}{\lambda^3}
\frac{R_0^4\cos i}{bD^2c^2}
\left(\frac{\lambda kT_B}{hc}\right)^{4/b}
I_4(b)
\left(1-\epsilon^2\right)
\left(1+\frac12\sin^2i\right).
}
$$

三个矩都带有共同的绝对归一化，但在归一化矩中消去。

### 4.5 归一化高阶原点矩

由 $m_n=M_n/M_0$ 得

$$
\boxed{
m_n(\lambda)
=
(\tau_\lambda^*)^n
\mathcal A_n(i)
\frac{I_{n+2}(b)}{I_2(b)}
\frac{1-\epsilon^{(n+2)/2}}{1-\epsilon}.
}
$$

所以

$$
\boxed{
m_n(\lambda)\propto\lambda^{n/b}.
}
$$

例如

$$
m_3
=
(\tau_\lambda^*)^3
\left(1+\frac32\sin^2i\right)
\frac{I_5}{I_2}
\frac{1-\epsilon^{5/2}}{1-\epsilon},
$$

$$
m_4
=
(\tau_\lambda^*)^4
\left(1+3\sin^2i+\frac38\sin^4i\right)
\frac{I_6}{I_2}
(1+\epsilon+\epsilon^2).
$$

标准薄盘 $b=3/4$ 时，$m_1,m_2,m_3,m_4$ 的理论波长指数依次为

$$
\frac43,
\qquad
\frac83,
\qquad
4,
\qquad
\frac{16}{3}.
$$

必须区分 $M_2$ 与 $m_2$：在 $B_\nu$ 表示下

$$
M_2\propto\lambda^{4/b-3},
\qquad
m_2\propto\lambda^{2/b}.
$$

因此 $b=3/4$ 时，未归一化 $M_2$ 的指数是 $7/3$，而归一化原点二阶矩及中心二阶矩的指数才是 $8/3$。

---

## 5. 从原始矩得到中心二阶矩

质心为

$$
\bar\tau_\lambda
\equiv
\frac{M_1}{M_0}
=
\tau_\lambda^*
\frac{I_3}{I_2}
\frac{1-\epsilon^{3/2}}{1-\epsilon}.
$$

原点二阶矩为

$$
\begin{aligned}
\left\langle\tau^2\right\rangle_\lambda
&\equiv
\frac{M_2}{M_0}\\
&=
(\tau_\lambda^*)^2
\left(1+\frac12\sin^2i\right)
\frac{I_4}{I_2}
\frac{1-\epsilon^2}{1-\epsilon}.
\end{aligned}
$$

因为

$$
\frac{1-\epsilon^2}{1-\epsilon}=1+\epsilon,
$$

所以

$$
\left\langle\tau^2\right\rangle_\lambda
=
(\tau_\lambda^*)^2
\left(1+\frac12\sin^2i\right)
\frac{I_4}{I_2}(1+\epsilon).
$$

中心二阶矩为

$$
\sigma_{\tau,\lambda}^2
\equiv
\left\langle
(\tau-\bar\tau_\lambda)^2
\right\rangle
=
\left\langle\tau^2\right\rangle_\lambda
-\bar\tau_\lambda^2.
$$

最终得到

$$
\boxed{
\sigma_{\tau,\lambda}^2
=
(\tau_\lambda^*)^2
\left[
\left(1+\frac12\sin^2 i\right)
\frac{I_4}{I_2}(1+\epsilon)
-
\left(
\frac{I_3}{I_2}
\frac{1-\epsilon^{3/2}}{1-\epsilon}
\right)^2
\right].
}
$$

四个容易出错的地方是：

1. 倾角因子只乘原点二阶矩的 $I_4/I_2$ 项；
2. 它不能乘整个中心二阶矩；
3. 均值中的 $I_3/I_2$ 和温度差分因子必须一起平方；
4. 原点二阶矩使用 $I_4/I_2$，不是 $I_4/I_3$。

---

## 6. 径向展宽与倾角展宽

定义差分响应加权平均

$$
\langle g(R)\rangle_W
\equiv
\frac{
\int_0^\infty g(R)R\Delta B_\nu(R)\,dR
}{
\int_0^\infty R\Delta B_\nu(R)\,dR
}.
$$

则

$$
\bar\tau=\frac{\langle R\rangle_W}{c},
$$

$$
\left\langle\tau^2\right\rangle
=
\left(1+\frac12\sin^2i\right)
\frac{\langle R^2\rangle_W}{c^2}.
$$

因此

$$
\boxed{
\sigma_\tau^2
=
\underbrace{
\frac{
\langle R^2\rangle_W-\langle R\rangle_W^2
}{c^2}
}_{\text{径向响应权重造成的展宽}}
+
\underbrace{
\frac{\sin^2i}{2}
\frac{\langle R^2\rangle_W}{c^2}
}_{\text{同一圆环近端—远端造成的倾角展宽}}.
}
$$

即使 $i=0$，转移函数仍有非零宽度，因为一个波段并不只来自单一半径。

---

## 7. $b=3/4$ 的准确数值系数

标准薄盘取

$$
b=\frac34.
$$

此时

$$
I_2
=
\Gamma\!\left(\frac83\right)
\zeta\!\left(\frac83\right)
=
1.93216160908924,
$$

$$
I_3
=
\Gamma(4)\zeta(4)
=
\frac{\pi^4}{15}
=
6.49393940226683,
$$

$$
I_4
=
\Gamma\!\left(\frac{16}{3}\right)
\zeta\!\left(\frac{16}{3}\right)
=
41.2765366414990.
$$

所以

$$
\frac{I_3}{I_2}
=
3.36097113808605,
$$

$$
\left(\frac{I_3}{I_2}\right)^2
=
11.2961269910474,
$$

$$
\frac{I_4}{I_2}
=
21.3628800237655,
$$

$$
\frac{I_4I_2}{I_3^2}
=
1.89116854304987.
$$

中心二阶矩为

$$
\boxed{
\frac{\sigma_{\tau,\lambda}^2}
{(\tau_\lambda^*)^2}
=
21.3628800237655
(1+\epsilon)
\left(1+\frac12\sin^2i\right)
-
11.2961269910474
\left(
\frac{1-\epsilon^{3/2}}{1-\epsilon}
\right)^2.
}
$$

其中

$$
\epsilon
=
\left(\frac{T_F}{T_B}\right)^{8/3}.
$$

---

## 8. 带物理单位的前系数

取

$$
R_0=1\ {\rm light\!-\!day},
\qquad
\lambda_4
\equiv
\frac{\lambda}{10^4\ {\rm \mathring A}},
\qquad
T_4
\equiv
\frac{T_B}{10^4\ {\rm K}}.
$$

则

$$
\boxed{
\tau_\lambda^*
=
0.615661600620351\ {\rm d}\,
\lambda_4^{4/3}T_4^{4/3}.
}
$$

令

$$
\mathcal S
\equiv
\lambda_4^{4/3}T_4^{4/3},
$$

便有

$$
\boxed{
\frac{\sigma_{\tau,\lambda}^2}{{\rm d}^2}
=
\mathcal S^2
\left[
8.09736909230159
(1+\epsilon)
\left(1+\frac12\sin^2i\right)
-
4.28167501096599
\left(
\frac{1-\epsilon^{3/2}}{1-\epsilon}
\right)^2
\right].
}
$$

若 $R_0\neq1$ light-day，右侧再乘

$$
\left(
\frac{R_0}{1\ {\rm light\!-\!day}}
\right)^2.
$$

---

## 9. $T_B/T_F\to1$ 的线性极限

当

$$
\epsilon\to1,
$$

有

$$
\frac{1-\epsilon^{3/2}}{1-\epsilon}
\longrightarrow
\frac32,
\qquad
1+\epsilon\longrightarrow2.
$$

因此

$$
\boxed{
\frac{\sigma_{\tau,\lambda}^2}
{(\tau_\lambda^*)^2}
=
17.3094743176742
+
21.3628800237655\sin^2i.
}
$$

带单位形式为

$$
\boxed{
\frac{\sigma_{\tau,\lambda}^2}{{\rm d}^2}
=
\mathcal S^2
\left[
6.56096940992971
+
8.09736909230159\sin^2i
\right].
}
$$

同一极限下，

$$
\bar\tau_\lambda
=
5.04145670712907\,\tau_\lambda^*
=
3.10383130576929\ {\rm d}\,\mathcal S.
$$

数值实现中不应直接计算两个接近零的量。令

$$
\eta_\epsilon=\log\epsilon,
$$

则可稳定地计算

$$
\frac{1-\epsilon^a}{1-\epsilon}
=
\frac{
\operatorname{expm1}(a\eta_\epsilon)
}{
\operatorname{expm1}(\eta_\epsilon)
}.
$$

在 $|\eta_\epsilon|$ 很小时，直接切换到解析极限或级数展开。

---

## 10. 原文 $0.62\ {\rm d}$ 系数的核验

当

$$
R_0=1\ {\rm light\!-\!day},
\quad
\lambda=10^4\ {\rm \mathring A},
\quad
T_B=10^4\ {\rm K},
$$

有

$$
\tau_\lambda^*
=
0.615661600620351\ {\rm d}.
$$

但是质心公式还必须乘

$$
\frac{I_3}{I_2}
=
3.36097113808605.
$$

因此，由 Cackett 的符号公式及直接径向积分得到的、尚未乘亮暗温度因子的质心前系数是

$$
\boxed{
0.615661600620351
\times
3.36097113808605
=
2.06922087051286\ {\rm d}.
}
$$

原文 Eq. (13) 的 $0.62\ {\rm d}$ 对应的是 $\tau_\lambda^*$，没有包含其 Eq. (11) 中的 $I_3/I_2$。在线性极限还要再乘 $3/2$，得到

$$
3.10383130576929\ {\rm d}.
$$

后续代码不应照抄 Eq. (13) 的 $0.62\ {\rm d}$ 作为质心前系数，而应从物理常数和 $I_n$ 重新计算。

---

## 11. 消去尺度后的倾角约束

定义

$$
u_\lambda
\equiv
\frac{\sigma_{\tau,\lambda}^2}
{\bar\tau_\lambda^2},
$$

$$
C_b
\equiv
\frac{I_4I_2}{I_3^2},
$$

$$
G(\epsilon)
\equiv
\frac{
(1+\epsilon)(1-\epsilon)^2
}{
(1-\epsilon^{3/2})^2
}.
$$

则所有 $R_0,T_B,\lambda$ 的尺度消掉：

$$
\boxed{
u_\lambda
=
C_bG(\epsilon)
\left(1+\frac12\sin^2i\right)-1.
}
$$

所以

$$
\boxed{
\sin^2i
=
2\left[
\frac{1+u_\lambda}
{C_bG(\epsilon)}
-1
\right].
}
$$

倾角为

$$
\boxed{
i
=
\arcsin
\sqrt{
2\left[
\frac{1+u_\lambda}
{C_bG(\epsilon)}
-1
\right]
}.
}
$$

对于 $b=3/4$，

$$
C_b=1.89116854304987.
$$

在线性极限，

$$
G(\epsilon)\to\frac89,
$$

所以

$$
\boxed{
u_\lambda
=
0.681038704933218
+
0.840519352466609\sin^2i.
}
$$

反解为

$$
\boxed{
\sin^2i
=
\frac{
u_\lambda-0.681038704933218
}{
0.840519352466609
}.
}
$$

模型允许的范围是

$$
\boxed{
0.681038704933218
\leq
u_\lambda
\leq
1.52155805739983.
}
$$

若后验主要落在这个范围之外，不应简单截断到 $i=0^\circ$ 或 $90^\circ$，而应检查盘边界、bandpass、非盘残留、相对核映射以及 MICA 核族投影误差。

### 11.1 由 W2 锚定拟合系数反演倾角

若各波段绝对盘核属于同一自相似族，并且 MICA 盘相对核满足精确卷积闭合，则一阶矩和中心二阶矩差可以联合写成

$$
\Delta\mu_\lambda
=
\texttt{tau0}
\left[
\left(\frac{\lambda}{\lambda_0}\right)^{1/b}-1
\right],
$$

$$
\Delta V_\lambda
=
\texttt{moment20}
\left[
\left(\frac{\lambda}{\lambda_0}\right)^{2/b}-1
\right].
$$

两个拟合系数的物理含义是

$$
\boxed{
\texttt{tau0}=\mu_0,
\qquad
\texttt{moment20}=V_0.
}
$$

这里的 `tau0` 和 `moment20` 是由其他波段相对于 W2 的差值外推得到的 W2 绝对盘核质心和方差尺度；W2 相对自身的核仍然是零质心、零方差的恒等 $\delta$ 核。

因此用于倾角反演的无尺度量为

$$
\boxed{
u_0
=
\frac{\texttt{moment20}}{\texttt{tau0}^2}.
}
$$

代入本节的一般关系，

$$
\boxed{
i
=
\arcsin
\sqrt{
2\left[
\frac{1+\texttt{moment20}/\texttt{tau0}^2}
{C_bG(\epsilon)}-1
\right]
}.
}
$$

对于 $b=3/4$ 且 $\epsilon\to1$，

$$
\boxed{
i
=
\arcsin
\sqrt{
\frac{
\texttt{moment20}/\texttt{tau0}^2
-0.681038704933218
}{0.840519352466609}
}.
}
$$

倾角不是由指数 $\gamma$ 单独决定的。理论指数关系

$$
\beta=\frac1b,
\qquad
\gamma=\frac2b=2\beta
$$

约束温度律和矩闭合。标准薄盘 $b=3/4$ 给出 $\gamma=8/3$，slim-like 的 $b=1/2$ 给出 $\gamma=4$；倾角来自一阶与二阶拟合系数的比值，而不是来自这两个指数中的某一个。

### 11.2 更适合贝叶斯前向模型的参数化

与其分别拟合两个系数后再用边缘中位数反解，更自然的做法是直接把 `moment20` 写成派生量：

$$
\boxed{
\texttt{moment20}
=
\texttt{tau0}^2
\left[
C_bG(\epsilon)
\left(1+\frac12\sin^2i\right)-1
\right].
}
$$

然后联合采样

$$
(\texttt{tau0},b,\epsilon,i,\ldots).
$$

若仍采用两阶段后验，则必须对每个联合 posterior draw 计算

$$
u_s
=
\frac{\texttt{moment20}_s}{\texttt{tau0}_s^2},
$$

并令 $C_b$ 随 $b_s$ 更新。不能用两个边缘中位数的比值代替联合传播。根号内若落在 $[0,1]$ 之外，不应裁剪，而应视为该样本与理想 Cackett 模型不相容。

上述系数反演要求 W2 在物理模型中是同族有限宽盘核。若严格把 W2 的绝对物理核设为 shifted $\delta$，则 $V_0=0$，带 $-1$ 的 `moment2_model` 和这一倾角反演均失效；两种假设的详细区别见第 12.4 节。

---

## 12. W2 参考波段下的 MICA 相对核

### 12.1 绝对物理核、精确相对算子和 MICA 参数核

令不可观测的真实中心驱动为 $x(t)$，W2 记作参考波段 $0$。以下 $F_0$ 指去除测量噪声后的 latent W2 光变，而不是把含噪观测序列逐点当成精确驱动。绝对物理关系为

$$
F_0=\Psi_0*x,
\qquad
F_\lambda=\Psi_\lambda*x.
$$

MICA 把观测 W2 当作 driver，拟合

$$
F_\lambda
=
K_{\lambda|0}*F_0.
$$

若这一光变关系对所有容许驱动都成立，或者真实驱动在所讨论频段具有非零谱功率，并且共同驱动、线性时不变与模型精确等条件成立，则核必须满足

$$
\boxed{
\Psi_\lambda
=
K_{\lambda|0}*\Psi_0.
}
$$

频域中即

$$
\boxed{
\widetilde K_{\lambda|0}(\omega)
=
\frac{\widetilde\Psi_\lambda(\omega)}
{\widetilde\Psi_0(\omega)}.
}
$$

该比值只在 $\widetilde\Psi_0(\omega)\neq0$ 且驱动对该频率有辨识力的区域有意义。因此 MICA 的对象不是绝对核 $\Psi_\lambda$，而是以 W2 为参考的相对反卷积算子。精确 $K$ 未必是正的、因果的或 Gamma 形；MICA 的正 Gamma/双 Gamma 核是对它的受限参数化投影。

### 12.2 卷积矩关系的直接证明

设

$$
h(t)=(f*g)(t)
=
\int f(u)g(t-u)\,du,
$$

并定义

$$
F_n=\int u^nf(u)\,du,
\qquad
G_n=\int v^ng(v)\,dv,
\qquad
H_n=\int t^nh(t)\,dt.
$$

令 $v=t-u$，即 $t=u+v$。则零阶矩为

$$
\boxed{
H_0=F_0G_0.
}
$$

一阶矩为

$$
\begin{aligned}
H_1
&=
\iint(u+v)f(u)g(v)\,du\,dv\\
&=
F_1G_0+F_0G_1.
\end{aligned}
$$

二阶原点矩为

$$
\begin{aligned}
H_2
&=
\iint(u+v)^2f(u)g(v)\,du\,dv\\
&=
F_2G_0+2F_1G_1+F_0G_2.
\end{aligned}
$$

定义归一化质心和原点二阶矩

$$
\mu_f=\frac{F_1}{F_0},
\qquad
q_f=\frac{F_2}{F_0},
$$

其余核类似。由上式得到

$$
\boxed{
\mu_h=\mu_f+\mu_g,
}
$$

$$
\boxed{
q_h=q_f+2\mu_f\mu_g+q_g.
}
$$

中心二阶矩 $V=q-\mu^2$ 因而满足

$$
\begin{aligned}
V_h
&=
q_f+2\mu_f\mu_g+q_g-(\mu_f+\mu_g)^2\\
&=
(q_f-\mu_f^2)+(q_g-\mu_g^2),
\end{aligned}
$$

即

$$
\boxed{
V_h=V_f+V_g.
}
$$

一般的归一化原点矩满足

$$
\boxed{
m_{n,h}
=
\sum_{j=0}^{n}
\binom nj
m_{j,f}m_{n-j,g}.
}
$$

所以高阶原点矩包含所有低阶交叉项；能够在卷积下直接相加的是累积量：

$$
\boxed{
\kappa_n(h)=\kappa_n(f)+\kappa_n(g).
}
$$

### 12.3 相对核的面积、质心和方差

把上一小节取为 $h=\Psi_\lambda$、$f=K_{\lambda|0}$、$g=\Psi_0$，得到

$$
\boxed{
M_{0,K}
=
\frac{M_{0,\lambda}}{M_{0,0}},
}
$$

$$
\boxed{
\mu_K
=
\mu_\lambda-\mu_0,
}
$$

$$
\boxed{
V_K
=
V_\lambda-V_0.
}
$$

但相对核的原点二阶矩不是两个绝对原点二阶矩之差，而是

$$
\boxed{
q_K
=
q_\lambda-q_0
-2\mu_0(\mu_\lambda-\mu_0).
}
$$

等价地，

$$
q_K
=
(V_\lambda-V_0)
+(\mu_\lambda-\mu_0)^2.
$$

这些等式针对完整精确相对核。数据 gap、噪声、DRW 先验和正 Gamma 核族会使 MICA 后验成为该算子的模型依赖近似。

精确相对算子可以带负瓣，因此这里的 $V_K$ 是二阶累积量，不保证像正概率分布的方差那样非负。MICA 的正 Gamma 参数化则把 $V_K$ 限制为非负；这是“精确相对算子”与“MICA 最佳拟合核”之间可能出现系统偏差的一处来源。

### 12.4 W2 为 shifted $\delta$ 与 W2 为有限宽盘核

若严格假设

$$
\Psi_0(\tau)
=
A_{\rm W2}\delta(\tau-\tau_{\rm W2}),
$$

则

$$
M_{0,0}=A_{\rm W2},
\qquad
\mu_0=\tau_{\rm W2},
\qquad
V_0=0.
$$

卷积关系给出

$$
M_{0,K}=\frac{M_{0,\lambda}}{A_{\rm W2}},
\qquad
\mu_K=\mu_\lambda-\tau_{\rm W2},
\qquad
V_K=V_\lambda.
$$

若对 $\lambda\neq\lambda_0$ 的目标波段，$B_\nu$ 下的绝对盘核满足

$$
M_{0,\lambda}\propto\lambda^{2/b-3},
\qquad
\mu_\lambda\propto\lambda^{1/b},
\qquad
V_\lambda\propto\lambda^{2/b},
$$

并令 $A_{\rm W2}=M_{0}(\lambda_0)$、$\tau_{\rm W2}=\mu(\lambda_0)$，则

$$
\boxed{
M_{0,K}
=
\left(\frac{\lambda}{\lambda_0}\right)^{2/b-3},
}
$$

$$
\boxed{
\mu_K
=
\tau_{\rm W2}
\left[
\left(\frac{\lambda}{\lambda_0}\right)^{1/b}-1
\right],
}
$$

再定义目标盘族外推到参考波长时本来应有的方差尺度

$$
V_{{\rm fam},0}\equiv C_V\lambda_0^{2/b}.
$$

注意它不是 shifted-$\delta$ W2 的实际方差；后者严格为零。此时

$$
\boxed{
V_K
=
V_{{\rm fam},0}
\left(\frac{\lambda}{\lambda_0}\right)^{2/b},
}
$$

没有 $-1$。平移 $\delta$ 改变质心但不增加方差。

这里的 shifted-$\delta$ 是一种混合近似：它让 W2 的面积和质心等于同族盘模型在 $\lambda_0$ 的值，却人为把 W2 的宽度压到零。因此它有意放弃了同族盘模型在 $\lambda_0$ 处的非零方差，不能再同时要求方差满足同族锚定差分律。特别地，$V_K\propto(\lambda/\lambda_0)^{2/b}$ 只可用于 $\lambda\neq\lambda_0$ 的目标波段；在参考点必须单独定义 W2 相对自身的恒等核，其方差为零，否则该混合近似在 $\lambda=\lambda_0$ 会自相矛盾。

只要精确卷积闭合成立，任意有限宽参考核都满足 $V_K=V_\lambda-V_0$。进一步把 W2 视为同一 Cackett 自相似族中的有限宽核，并满足

$$
V_0=C_V\lambda_0^{2/b},
$$

才有

$$
\boxed{
V_K
=
V_\lambda-V_0
=
V_0
\left[
\left(\frac{\lambda}{\lambda_0}\right)^{2/b}-1
\right].
}
$$

令 $r_\lambda=\lambda/\lambda_0$，有限宽同族 W2 假设下的三条锚定关系可统一写成

$$
\boxed{
M_{0,K}=r_\lambda^{2/b-3},
\qquad
\mu_K=\mu_0\left(r_\lambda^{1/b}-1\right),
\qquad
V_K=V_0\left(r_\lambda^{2/b}-1\right).
}
$$

其中只有零阶矩是面积比；一阶和中心二阶矩才是累积量差。这三者不能机械地套用同一个“减一”形式。

两套假设的区别为：

| W2 的物理假设 | $M_{0,K}$ | $\mu_K$ | $V_K$ |
|---|---|---|---|
| shifted $\delta$ | 面积比，无 $-1$ | 质心差，有 $-1$ | 目标绝对方差，无 $-1$ |
| 同族有限宽盘核 | 面积比，无 $-1$ | 质心差，有 $-1$ | 方差差，有 $-1$ |

因此，严格的 shifted-$\delta$ W2 假设与当前带 $-1$ 的 `moment2_model` 不是同一套物理假设。

### 12.5 由单个相对核恢复绝对无尺度矩比

若采用有限宽同族 W2 假设，定义

$$
\rho
\equiv
\left(\frac{\lambda}{\lambda_0}\right)^{1/b}.
$$

由于

$$
\mu_K=\mu_0(\rho-1),
\qquad
V_K=V_0(\rho^2-1),
$$

绝对核的无尺度比 $u=V_0/\mu_0^2$ 为

$$
\boxed{
u
=
\frac{V_K}{\mu_K^2}
\frac{\rho-1}{\rho+1}.
}
$$

因此不能把相对核的 $V_K/\mu_K^2$ 直接代入倾角公式。

---

## 13. 固定 $k=2$ Gamma MICA 核与代码含义

### 13.1 单 Gamma 分量

MICA 的 shifted-Gamma 分量为

$$
\Psi_k(\tau)
=
\frac{A_k}{w_k^2}
(\tau-c_k)
\exp\!\left[-\frac{\tau-c_k}{w_k}\right]
\Theta(\tau-c_k),
$$

其中 $A_k$ 是积分面积，$c_k$ 是支撑起点而不是质心，$w_k$ 是尺度。在当前正响应配置下，MICA 原始后验中实际存储的是 $\log A_k,c_k,\log w_k$；若允许负响应，振幅参数化需要另行处理。

本节中的 $k=0,1$ 是 Gamma 组分编号；第 12 节中的参考波段下标 $0$ 则表示 W2，两者不是同一个“0”。

未归一化矩为

$$
M_{0,k}=A_k,
$$

$$
M_{1,k}=A_k(c_k+2w_k),
$$

$$
M_{2,k}
=
A_k(c_k^2+4c_kw_k+6w_k^2).
$$

所以

$$
\boxed{
\mu_k=c_k+2w_k,
}
$$

$$
\boxed{
q_k=c_k^2+4c_kw_k+6w_k^2,
}
$$

$$
\boxed{
V_k=2w_k^2.
}
$$

单分量中心二阶矩只由 $w_k$ 决定；原点二阶矩需要 $c_k,w_k$；未归一化 $M_2$ 还需要 $A_k$。

任意非负整数阶原点矩都有通式

$$
\boxed{
M_{n,k}
=
A_k
\sum_{j=0}^{n}
\binom nj
c_k^{n-j}(j+1)!\,w_k^j.
}
$$

相应的累积量为

$$
\boxed{
\kappa_{1,k}=c_k+2w_k,
\qquad
\kappa_{n,k}=2(n-1)!\,w_k^n,
\quad n\geq2.
}
$$

### 13.2 双 Gamma 总核

定义面积分数和分量质心

$$
f_k=\frac{A_k}{A_0+A_1},
\qquad
\mu_k=c_k+2w_k.
$$

总质心为

$$
\boxed{
\mu_{\rm total}
=
\sum_{k=0}^1f_k\mu_k.
}
$$

总中心二阶矩为

$$
\boxed{
V_{\rm total}
=
\sum_{k=0}^1
f_k
\left[
2w_k^2+(\mu_k-\mu_{\rm total})^2
\right].
}
$$

对于两个组分，也可以写成

$$
\boxed{
V_{\rm total}
=
2f_0w_0^2+2f_1w_1^2
+f_0f_1(\mu_1-\mu_0)^2.
}
$$

最后一项是两个分量质心分离带来的组分间展宽。

更一般地，双组分总核的原点矩满足

$$
M_{n,{\rm total}}=\sum_{k=0}^{1}M_{n,k},
\qquad
m_{n,{\rm total}}=\sum_{k=0}^{1}f_km_{n,k}.
$$

但混合核的高阶累积量一般不能写成各分量累积量的面积加权平均；上面的全方差公式正是二阶时必须加入组分间展宽的例子。

### 13.3 当前代码实际拟合的量

MICA 本身拟合完整光变并采样 $A_k,c_k,w_k$，不直接拟合矩。`src/fit_tau.py` 的后处理对 Gamma 分量设置

$$
\mu_k=c_k+2w_k,
\qquad
V_k=2w_k^2,
$$

并用上一小节的全方差公式构造总核。随后保存和拟合的 `moment2` 是 $V$，即归一化中心二阶矩，不是 $M_2$ 或 $q$。

对 `comp0`，代码中的

$$
\texttt{moment2}=2w_0^2
$$

只有在额外满足分量级闭合

$$
\Psi_\lambda^{\rm disk}
=
K_{\lambda|{\rm W2}}^{\rm comp0}
*
\Psi_{\rm W2}^{\rm disk}
$$

时，才可解释为

$$
V_\lambda^{\rm disk}-V_{\rm W2}^{\rm disk}.
$$

卷积定理本身严格约束的是完整相对核。把 `comp0` 识别为盘还要求 W2 的成分足够干净、`comp1` 没有吞掉盘翼，并且正 Gamma 能充分表示真实盘相对算子。

### 13.4 单 Gamma 矩到倾角比值

若 `comp0` 的分量级闭合确实成立，而且采用有限宽同族 W2 假设，则

$$
\mu_K=c+2w,
\qquad
V_K=2w^2,
$$

从而

$$
\boxed{
u
=
\frac{2w^2}{(c+2w)^2}
\frac{\rho-1}{\rho+1}.
}
$$

该式应对完整 MICA 加权后验逐样本计算，不应只代入 $(c,w)$ 的边缘中位数。Mrk 817 中已经发现的 comp0/comp1 连续补偿和核族敏感性说明，这个分量级解释必须通过完整核闭合或注入恢复检验。

---

## 14. 有限盘和真实 bandpass

如果采用有限 $R_{\rm in},R_{\rm out}$ 或真实 bandpass，$I_n$ 闭式通常不再成立。定义

$$
\mathcal J_n
\equiv
\int_{R_{\rm in}}^{R_{\rm out}}
R^{n+1}\Delta\mathcal B(R)\,dR,
$$

其中 $\Delta\mathcal B(R)$ 已包含 bandpass 积分，则仍有

$$
\bar\tau
=
\frac1c\frac{\mathcal J_1}{\mathcal J_0},
$$

$$
\boxed{
\sigma_\tau^2
=
\frac1{c^2}
\left[
\left(1+\frac12\sin^2i\right)
\frac{\mathcal J_2}{\mathcal J_0}
-
\left(
\frac{\mathcal J_1}{\mathcal J_0}
\right)^2
\right].
}
$$

只要整圆盘的方位权重均匀，而且时延仍为

$$
\tau=\frac Rc(1+\sin i\cos\phi),
$$

那么

$$
1+\frac12\sin^2i
$$

这个倾角系数仍然精确。有限边界和 bandpass 只改变径向积分比值。

有限灯高、盘厚度、遮挡、方位不均匀或相对论增亮会改变方位角积分，届时不能继续使用这一简单倾角因子。

---

## 15. 数值实现与验证

### 15.1 frame

$$
\tau_{\rm obs}=(1+z)\tau_{\rm rest},
\qquad
\sigma_{\rm obs}^2=(1+z)^2\sigma_{\rm rest}^2.
$$

但

$$
\frac{\sigma_\tau^2}{\bar\tau^2}
$$

对统一时间伸缩不变，因此无尺度倾角反解式本身不受时间 frame 选择影响。

### 15.2 必须通过的检查

1. $i=0$ 时仍有非零径向方差；
2. $A_1/A_0=1$，质心不随 $i$ 改变；
3. $A_2/A_0=1+\sin^2i/2$；
4. $\epsilon\to1$ 时公式连续且有限；
5. 二维 $R,\phi$ 数值积分与解析 $M_0,M_1,M_2$ 一致；
6. 相对核满足累积量差关系；
7. 从完整离散理论核积分得到的矩与闭式一致。

独立离散积分已经检查

$$
r=0.5,\ 0.8,\ 0.99
$$

以及

$$
i=0^\circ,\ 30^\circ,\ 60^\circ,\ 85^\circ.
$$

质心、原点二阶矩和中心二阶矩与解析式的相对误差约为 $10^{-11}$。

---

## 16. 在贝叶斯模型中的接口

每个物理参数样本

$$
\theta=(\texttt{tau0},T_B,T_F,b,i,D,\ldots)
$$

这里 `tau0` 必须被定义为替代 $R_0T_B^{1/b}$ 这一物理尺度组合的采样参数，或者由 $R_0,T_B,b$ 派生；不能再把描述同一尺度的 `tau0`、$R_0$ 和 $T_B$ 全部作为彼此独立的自由参数。

至少输出

$$
\bar\tau_\lambda^{\rm th},
\qquad
\left\langle\tau^2\right\rangle_\lambda^{\rm th},
\qquad
\sigma_{\tau,\lambda}^{2,{\rm th}},
$$

以及相对于 W2 的

$$
\Delta\bar\tau_{\lambda|0}^{\rm th},
\qquad
\Delta V_{\lambda|0}^{\rm th}.
$$

若采用矩层 likelihood，应使用同一参数样本同时生成

$$
\Delta\mu_\lambda^{\rm th}
=
\texttt{tau0}
\left[
\left(\frac{\lambda}{\lambda_0}\right)^{1/b}-1
\right],
$$

$$
\Delta V_\lambda^{\rm th}
=
\texttt{tau0}^2
\left[
C_bG(\epsilon)
\left(1+\frac12\sin^2i\right)-1
\right]
\left[
\left(\frac{\lambda}{\lambda_0}\right)^{2/b}-1
\right].
$$

这样 $b$ 同时控制一、二阶波长指数，$i$ 只通过方差相对于质心平方的归一化进入。

若采用完整核 likelihood，物理闭合关系应写成

$$
\boxed{
\Psi_\lambda^{\rm th}
\approx
K_{\lambda|0}^{\rm MICA}
*
\Psi_0^{\rm th}.
}
$$

这比把 W2 在物理上直接设成 $\delta$ 更一致：W2 的绝对宽度保留在 $\Psi_0^{\rm th}$ 中，MICA 后验则提供相对核约束。若使用 `comp0` 而不是总核，必须把分量归属和盘翼泄漏列为模型假设或 discrepancy，而不能由编号自动确定物理身份。

建议保留两条互相校验的计算路径：

1. 用本笔记的闭式计算矩；
2. 从完整离散理论核直接积分计算矩。

只有两条路径在无限盘、单色配置下达到预设精度后，才将倾角作为自由参数进入 CDNest/MCMC。

正式 likelihood 中最好仍由完整核形状约束倾角；本笔记的中心二阶矩公式用于解析单元测试、初始化、先验范围检查和 posterior 的物理解读。

---

## 17. 参考资料

- Cackett, Horne & Winkler (2007), Testing thermal reprocessing in active galactic nuclei accretion discs: <https://academic.oup.com/mnras/article/380/2/669/1012849>
- Li, Wang & Bai (2016), A Non-parametric Approach to Constrain the Transfer Function in Reverberation Mapping: <https://arxiv.org/abs/1608.03741>
- MICA2 documentation: <https://mica2.readthedocs.io/en/latest/getting_started.html>
- 统计前向模型设计：notes/Cackett_MICA_两态差分贝叶斯前向模型统计框架.md
- 理论盘转移函数代码：test/disk_transfer/disktf_diff.py
- MICA Gamma 核实现：/home/leonhard/software/MICA2/plotfig.py
- 当前中心二阶矩实现：src/fit_tau.py

---

## 写作笔记

- 推导中心矩时先保留 $M_0,M_1,M_2$，最后计算 $M_2/M_0-(M_1/M_0)^2$，最不容易漏掉归一化或误把倾角乘到整个方差。
- 讨论倾角信息时，先写成“径向展宽 + 方位角展宽”，比只给最终闭式更容易说明 $1/2$ 系数来自哪里。
- 使用 MICA 相对核时，必须先区分绝对核矩与相对核累积量差；自检问题是：当前代入倾角公式的方差究竟属于哪个核。
- W2 锚定关系必须按矩的代数分别处理：零阶矩给面积比，一阶矩给质心差，中心二阶矩在有限宽 W2 假设下给方差差；不能统一套用同一个“减一”形式。
- 倾角来自 $\texttt{moment20}/\texttt{tau0}^2$ 的联合后验，而不是二阶波长指数本身；自检问题是：两个系数是否来自同一盘相对核并保留了协方差。
