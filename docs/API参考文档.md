# **API 参考文档**

---

## **1. `BaseHandAPI` - 基础手部数据**
### **功能**  
提供手部关键点坐标、角度、速度等基础数据访问接口。

### **核心方法**
#### **`hand_side()`**
- **功能**：返回手部是左手还是右手。
- **返回值**：`"left"` / `"right"` / `"Unknown"`。

#### **`img_pos(point_id: int = -1)`**
- **功能**：获取手部关键点的图像坐标（像素）。
- **参数**：
  - `point_id`: 关键点编号（0~20），默认返回所有关键点。
- **返回值**：`np.ndarray`（形状为 `(N, 2)` 或 `(2, )`）。

#### **`norm_pos(point_id: int = -1, copy: bool = True)`**
- **功能**：获取以手部矩形框为原点的归一化坐标（范围 `[0, 1]`）。
- **参数**：
  - `point_id`: 关键点编号（0~20），默认返回所有关键点。
  - `copy`: 是否返回副本（避免修改原始数据）。
- **返回值**：`np.ndarray`（形状为 `(N, 3)` 或 `(3, )`）。

#### **`wrist_npos(point_id: int = -1, copy: bool = True)`**
- **功能**：获取以手腕为原点的归一化坐标。
- **参数**：同 `norm_pos`。
- **返回值**：`np.ndarray`（形状为 `(N, 3)` 或 `(3, )`）。

#### **`finger_angle(point_id: int = -1, copy: bool = True)`**
- **功能**：获取手指关节的弯曲角度（弧度制）。
- **参数**：
  - `point_id`: 关键点编号（0~20），默认返回所有角度。
- **返回值**：`np.ndarray`（形状为 `(N, )` 或 `(1, )`）。

#### **`thumb_dist(other_point_id: int = -1, copy: bool = True)`**
- **功能**：获取大拇指与其他手指关键点的曼哈顿距离。
- **参数**：
  - `other_point_id`: 其他手指关键点编号（8/12/16/20），默认返回全部距离。
- **返回值**：`np.ndarray`（形状为 `(4, )` 或 `(1, )`）。

#### **`wrist_npos_velocity(point_id: int = -1)`**
- **功能**：获取关键点两帧之间的归一化坐标差值（速度）。
- **参数**：同 `norm_pos`。
- **返回值**：`np.ndarray`（形状为 `(N, 3)` 或 `(3, )`）。

#### **`angle_velocity(point_id: int = -1)`**
- **功能**：获取手指关节角度变化速度。
- **参数**：同 `finger_angle`。
- **返回值**：`np.ndarray`（形状为 `(N, )` 或 `(1, )`）。

---

## **2. `HandDataAPI` - 展平后的手部数据**
### **功能**  
将手部数据展平为一维数组，适用于模型训练等场景。

### **核心属性**
#### **`norm_pos`**
- **功能**：返回所有关键点的归一化坐标（63维）。
- **返回值**：`np.ndarray`（形状为 `(63, )`）。

#### **`norm_pos2DataFrame`**
- **功能**：返回归一化坐标的 `DataFrame` 格式数据。
- **返回值**：`pandas.DataFrame`。

#### **`finger`**
- **功能**：返回手指角度和指尖距离数据（63+4=67维）。
- **返回值**：`np.ndarray`（形状为 `(67, )`）。

#### **`all_data`**
- **功能**：返回所有手部数据的整合数组（63+67=130维）。
- **返回值**：`np.ndarray`（形状为 `(130, )`）。

#### **`all2DataFrame`**
- **功能**：返回所有手部数据的 `DataFrame` 格式。
- **返回值**：`pandas.DataFrame`。

---

## **3. `Gestrue` - 手指状态数据**
### **功能**  
基于 LightGBM模型的手指状态检测（伸出、触碰、并拢）。

### **核心方法**
#### **`fg_all_out`**
- **功能**：检测所有手指是否伸出。
- **返回值**：`np.ndarray`（形状为 `(5, )`，索引 0~4 分别对应拇指、食指、中指、无名指、小拇指）。

#### **`get_fg_out(idx: int)`**
- **功能**：检测指定手指是否伸出。
- **参数**：
  - `idx`: 手指索引（0~4）。
- **返回值**：`np.ndarray`（布尔值）。

#### **`fg_all_touch`**
- **功能**：检测所有手指指尖是否触碰大拇指。
- **返回值**：`np.ndarray`（形状为 `(4, )`，索引 0~3 分别对应食指、中指、无名指、小拇指）。

#### **`get_fg_touch(idx: int)`**
- **功能**：检测指定手指指尖是否触碰大拇指。
- **参数**：
  - `idx`: 手指索引（0~3）。
- **返回值**：`np.ndarray`（布尔值）。

#### **`fg_all_close`**
- **功能**：检测所有手指是否并拢。
- **返回值**：`np.ndarray`（形状为 `(4, )`，索引 0~3 分别对应拇指与食指、食指与中指等）。

#### **`get_fg_close(idx: int)`**
- **功能**：检测指定手指是否并拢。
- **参数**：
  - `idx`: 手指对索引（0~3）。
- **返回值**：`np.ndarray`（布尔值）。

---

## **4. `HandDrawing` - 手部数据可视化**
### **功能**  
提供手部关键点绘制工具，支持在图像上绘制手部结构。

### **核心方法**
#### **`draw_hand(image: np.ndarray)`**
- **功能**：在图像上绘制手部关键点和连线。
- **参数**：
  - `image`: 输入图像（`np.ndarray`）。
- **返回值**：绘制后的图像（`np.ndarray`）。

#### **`get_hand_img(padx: int = 0, pady: int = 0)`**
- **功能**：截取手部区域的图像。
- **参数**：
  - `padx`: X轴填充像素。
  - `pady`: Y轴填充像素。
- **返回值**：`np.ndarray`（截取的图像）。

#### **`draw_norm_hand(bg_img: HandImage)`**
- **功能**：在指定背景图上绘制归一化后的手部坐标。
- **参数**：
  - `bg_img`: 背景图像（`HandImage` 实例）。
- **返回值**：绘制后的背景图像（`np.ndarray`）。

---

## **5. `HandInputSchemes` - 手部交互方案**
### **功能**  
提供滑动、点击、摇杆等交互方案，支持自定义手势控制。

### **1. `FingerSwipeScheme` - 滑动检测**
#### **初始化参数**
```python
FingerSwipeScheme(
    hand_input: HandInput,
    hand_name: str,
    point_id: int = 8,
    swipe_velocity: float = 3,
    min_swipe_dist: float = 0.3,
    reset_time: float = 2
)
```
- **参数**：
  - `point_id`: 滑动检测的关键点（默认食指指尖）。
  - `swipe_velocity`: 滑动速度阈值。
  - `min_swipe_dist`: 最小滑动距离。
  - `reset_time`: 重置时间（秒）。

#### **核心属性**
- **`is_activate`**: 是否检测到滑动。
- **`start_point`**: 滑动起始位置。
- **`vector`**: 滑动向量。
- **`distance`**: 滑动距离。
- **`norm_vec`**: 归一化方向向量。

### **2. `FingertipButtonScheme` - 指尖按钮**
#### **初始化参数**
```python
FingertipButtonScheme(
    hand_input: HandInput,
    hand_name: str,
    finger_touch_idx: int = 0,
    short_press_time: float = 0.1,
    long_press_time: float = 0.3
)
```
- **参数**：
  - `finger_touch_idx`: 检测的指尖索引（默认食指）。
  - `short_press_time`: 短按时间阈值（秒）。
  - `long_press_time`: 长按时间阈值（秒）。

#### **核心属性**
- **`is_short_press`**: 是否短按。
- **`is_long_press`**: 是否长按。
- **`start_point`**: 按下起点位置。

### **3. `ThumbJoystickScheme` - 拇指摇杆**
#### **初始化参数**
```python
ThumbJoystickScheme(
    hand_input: HandInput,
    hand_name: str,
    finger_btn: FingertipButtonScheme
)
```
- **参数**：
  - `finger_btn`: 用于长按设置定点的按钮方案。

#### **核心属性**
- **`is_activate`**: 摇杆是否激活。
- **`fixed_point`**: 摇杆固定点位置。
- **`vector`**: 摇杆向量。