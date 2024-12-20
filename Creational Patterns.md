빌더 패턴(Builder Pattern)은 복잡한 객체를 단계별로 생성할 수 있도록 돕는 디자인 패턴입니다. 이 패턴은 객체의 생성과정을 분리하여, 동일한 생성 절차에서 서로 다른 표현을 만들어낼 수 있도록 설계되어 있습니다. 특히 객체의 생성 과정이 복잡하거나 다양한 구성 요소를 가진 객체를 생성할 때 유용합니다.

빌더 패턴을 활용하면 객체를 생성하는 동안 필요한 설정이나 매개변수를 세부 단계별로 분리할 수 있기 때문에, 코드의 가독성과 유연성이 향상됩니다.

### 빌더 패턴을 이용한 예시: 컴퓨터 조립

빌더 패턴을 사용하여 컴퓨터를 조립하는 과정을 예시로 들어보겠습니다. 컴퓨터는 CPU, RAM, 저장 장치, 그래픽 카드 등 다양한 구성 요소로 이루어져 있으며, 이러한 구성 요소는 선택적으로 설정될 수 있습니다.

### 1. 컴퓨터 클래스 (복잡한 객체)

```python
class Computer:
    def __init__(self):
        self.cpu = None
        self.ram = None
        self.storage = None
        self.gpu = None

    def __str__(self):
        return (f"Computer specs:\n"
                f"CPU: {self.cpu}\n"
                f"RAM: {self.ram}\n"
                f"Storage: {self.storage}\n"
                f"GPU: {self.gpu}")

```

### 2. 빌더 클래스 정의

빌더 클래스는 컴퓨터 객체를 생성하는 과정의 각 단계를 제공합니다.

```python
class ComputerBuilder:
    def __init__(self):
        self.computer = Computer()

    def add_cpu(self, cpu):
        self.computer.cpu = cpu
        return self  # 현재 인스턴스를 반환하여 메서드 체이닝을 가능하게 함

    def add_ram(self, ram):
        self.computer.ram = ram
        return self

    def add_storage(self, storage):
        self.computer.storage = storage
        return self

    def add_gpu(self, gpu):
        self.computer.gpu = gpu
        return self

    def build(self):
        return self.computer

```

### 3. 디렉터 클래스 (선택 사항)

디렉터 클래스는 객체 생성의 단계들을 지정된 순서에 따라 호출함으로써 빌더 패턴을 관리하는 역할을 합니다. 이는 빌더 패턴의 필수 요소는 아니지만, 객체 생성 절차를 캡슐화하고 재사용 가능하게 만들어줍니다.

```python
class Director:
    def __init__(self, builder):
        self._builder = builder

    def build_gaming_computer(self):
        return (self._builder
                .add_cpu("Intel Core i9")
                .add_ram("32GB DDR4")
                .add_storage("1TB NVMe SSD")
                .add_gpu("NVIDIA RTX 3080")
                .build())

    def build_office_computer(self):
        return (self._builder
                .add_cpu("Intel Core i5")
                .add_ram("16GB DDR4")
                .add_storage("512GB SSD")
                .build())

```

### 4. 클라이언트 코드 (컴퓨터 생성)

클라이언트 코드에서 빌더를 사용하여 컴퓨터 객체를 생성합니다.

```python
# 빌더 인스턴스 생성
builder = ComputerBuilder()

# 디렉터를 사용하여 컴퓨터 조립
director = Director(builder)
gaming_computer = director.build_gaming_computer()
office_computer = director.build_office_computer()

print(gaming_computer)
print("\n")
print(office_computer)

```

### 실행 결과

```yaml
Computer specs:
CPU: Intel Core i9
RAM: 32GB DDR4
Storage: 1TB NVMe SSD
GPU: NVIDIA RTX 3080

Computer specs:
CPU: Intel Core i5
RAM: 16GB DDR4
Storage: 512GB SSD
GPU: None

```

### 빌더 패턴의 핵심 포인트

- **복잡한 객체의 생성 과정을 단계별로 분리**: 빌더 패턴을 통해 복잡한 객체를 단계별로 생성할 수 있으므로, 생성 과정을 유연하게 관리할 수 있습니다.
- **메서드 체이닝을 통한 유연한 구성**: `add_cpu()`, `add_ram()` 등의 메서드를 체이닝 방식으로 호출할 수 있어 가독성이 높아지고 구성 방식이 유연해집니다.
- **다양한 객체 생성 가능**: 동일한 빌더를 사용하여 다양한 객체(예: 게이밍 컴퓨터, 오피스 컴퓨터 등)를 생성할 수 있습니다.
- **디렉터(선택 사항)**: 디렉터는 생성 과정의 순서를 관리하는 역할을 담당하며, 여러 유형의 객체 생성 절차를 재사용 가능하게 해줍니다.

### 빌더 패턴의 장단점

### 장점:

- **복잡한 객체 생성의 단순화**: 객체 생성 과정을 단계별로 나눌 수 있어 복잡한 객체의 생성이 더 간단해집니다.
- **유연성과 가독성**: 메서드 체이닝을 통해 객체 생성이 더 유연하고 가독성이 높아집니다.
- **다양한 표현 지원**: 동일한 빌더로 다양한 구성의 객체를 생성할 수 있습니다.

### 단점:

- **클래스 증가**: 빌더와 디렉터를 추가로 정의해야 하므로, 코드의 구조가 복잡해지고 클래스의 수가 증가할 수 있습니다.
- **간단한 객체에는 오버헤드**: 빌더 패턴은 복잡한 객체에 적합하며, 간단한 객체에는 불필요한 오버헤드가 될 수 있습니다.

### 결론

빌더 패턴은 복잡한 객체의 생성 과정을 세분화하고 단계별로 관리해야 할 때 매우 유용합니다. 이 패턴을 통해 객체 생성 과정의 유연성과 가독성을 높일 수 있으며, 다양한 표현 방식을 지원하여 다양한 구성의 객체를 쉽게 생성할 수 있습니다.

### ChatGPT의 말:

ChatGPT

프로토타입(Prototype) 패턴은 이미 생성된 객체를 복제(clone)하여 새로운 객체를 생성하는 디자인 패턴입니다. 객체를 복사하는 과정을 통해 새로운 객체를 생성하므로, 직접 생성하는 것보다 시간과 비용을 절약할 수 있습니다. 특히, 객체 생성에 복잡한 초기화 작업이 필요한 경우 또는 동일한 구조의 객체를 반복적으로 생성해야 하는 경우에 유용합니다.

### 핵심 아이디어

- **객체의 복제**: 프로토타입 패턴에서는 새로운 객체를 만들기 위해 기존 객체를 복사(clone)합니다.
- **초기화 비용 감소**: 복잡한 초기화 과정을 거치지 않고 객체를 복제함으로써 생성 비용을 줄입니다.
- **런타임에 객체의 생성 유형 결정**: 런타임에 복제할 프로토타입 객체를 결정할 수 있어 유연성이 높습니다.

### 예시: 문서 편집기에서 도형 복제

문서 편집기에서 여러 종류의 도형(예: 원, 사각형)을 생성하고, 복사하여 새로운 도형을 만드는 경우를 예시로 살펴보겠습니다.

### 1. 프로토타입 인터페이스 정의

먼저, 복제 기능을 제공하는 프로토타입 인터페이스를 정의합니다.

```python
from abc import ABC, abstractmethod
import copy

# 프로토타입 인터페이스 (추상 클래스)
class Shape(ABC):
    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def draw(self):
        pass

```

### 2. 구체적인 프로토타입 클래스 정의

이제 `Shape`를 상속하여 원(`Circle`)과 사각형(`Rectangle`)을 구현하고, 각 클래스에서 `clone()` 메서드를 정의합니다.

```python
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def clone(self):
        return copy.deepcopy(self)

    def draw(self):
        print(f"Drawing a circle with radius {self.radius}")

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def clone(self):
        return copy.deepcopy(self)

    def draw(self):
        print(f"Drawing a rectangle with width {self.width} and height {self.height}")

```

- `copy.deepcopy(self)`: `deepcopy`를 사용하여 객체를 완전하게 복사합니다. 이 방법은 객체가 참조하는 모든 속성들을 함께 복제합니다.

### 3. 클라이언트 코드

클라이언트 코드에서는 원본 객체를 생성한 후, `clone()` 메서드를 사용하여 객체를 복제합니다.

```python
if __name__ == "__main__":
    # 원본 객체 생성
    original_circle = Circle(5)
    original_rectangle = Rectangle(10, 20)

    # 객체 복제
    cloned_circle = original_circle.clone()
    cloned_rectangle = original_rectangle.clone()

    # 복제된 객체 사용
    original_circle.draw()     # Drawing a circle with radius 5
    cloned_circle.draw()       # Drawing a circle with radius 5

    original_rectangle.draw()  # Drawing a rectangle with width 10 and height 20
    cloned_rectangle.draw()    # Drawing a rectangle with width 10 and height 20

```

### 프로토타입 패턴의 장단점

### 장점

- **객체 생성 비용 감소**: 객체를 직접 생성하는 대신 복제하는 방식으로 초기화 비용을 절약할 수 있습니다.
- **복잡한 객체의 생성 간소화**: 복잡한 초기화 과정이 필요한 객체를 간단하게 생성할 수 있습니다.
- **유연성**: 런타임에 객체의 생성 유형을 동적으로 결정할 수 있어 유연성이 높습니다.

### 단점

- **객체 복제의 복잡성**: 깊은 복사와 얕은 복사에 대한 이해가 필요하며, 복제 과정이 복잡해질 수 있습니다. 특히, 객체의 참조 관계가 복잡할 경우 제대로 복제하지 않으면 원본 객체와 복제된 객체 간에 의도치 않은 상호작용이 발생할 수 있습니다.
- **클론 메서드 구현 부담**: 모든 클래스에서 `clone()` 메서드를 구현해야 하므로, 클래스의 수가 많아질수록 코드 작성의 부담이 증가합니다.

### 언제 사용해야 하는가?

- **객체 생성 비용이 높거나 복잡한 초기화가 필요한 경우**: 객체를 생성하는 데 시간이 많이 소요되거나 복잡한 설정이 필요한 경우 프로토타입 패턴을 사용하면 효율적입니다.
- **동일한 구조의 객체를 여러 개 생성해야 하는 경우**: 반복적으로 동일한 구조의 객체를 생성해야 한다면 프로토타입 패턴을 사용해 객체를 복제하는 것이 효율적입니다.
- **객체의 생성 방식이 런타임에 결정되어야 하는 경우**: 런타임에 객체의 유형을 결정해야 할 때, 프로토타입 패턴을 사용하면 쉽게 객체를 생성할 수 있습니다.

### 결론

프로토타입 패턴은 기존 객체를 복제하여 새로운 객체를 생성하는 효과적인 방법을 제공합니다. 특히, 객체 생성이 복잡하거나 비용이 많이 드는 경우에 사용하면 유용합니다. 하지만 객체 복제의 복잡성에 주의해야 하며, 올바른 깊은 복사와 얕은 복사를 이해하고 구현하는 것이 중요합니다.

생성 디자인 패턴(Creational Design Patterns)은 객체 생성 과정을 캡슐화하고 추상화하여, 객체 생성 로직을 클라이언트 코드와 분리하는 데 사용됩니다. 이를 통해 코드의 유연성과 확장성을 높이고, 객체 생성의 복잡성을 관리할 수 있습니다. 대표적인 생성 디자인 패턴에는 다음과 같은 5가지가 있습니다.

### 1. **싱글톤 패턴 (Singleton Pattern)**

### **개념**

싱글톤 패턴은 클래스의 인스턴스가 오직 하나만 존재하도록 보장하며, 전역적으로 접근할 수 있는 단일 인스턴스를 제공합니다.

### **언제 사용하나?**

- 시스템 전체에서 공통된 리소스를 관리할 때 (예: 데이터베이스 연결, 설정 관리, 로깅 등)

### **예시**

- **데이터베이스 연결 풀**: 애플리케이션에서 데이터베이스에 연결할 때, 하나의 연결 풀을 여러 곳에서 공유하도록 하기 위해 싱글톤 패턴을 사용하면 자원 낭비를 막을 수 있습니다.

### 2. **팩토리 메서드 패턴 (Factory Method Pattern)**

### **개념**

팩토리 메서드 패턴은 객체 생성을 서브클래스에 위임하여, 객체 생성의 구체적인 로직을 클라이언트 코드와 분리하는 패턴입니다.

### **언제 사용하나?**

- 객체 생성 로직이 복잡하거나 변경될 가능성이 있을 때
- 클라이언트 코드가 생성할 객체의 타입에 대해 알지 못하거나 알 필요가 없을 때

### **예시**

- **문서 편집기**: 다양한 파일 형식(예: PDF, Word, Excel 등)을 처리하는 문서 편집기에서 각 형식에 맞는 문서 객체를 생성할 때 사용합니다. 파일 형식에 따라 문서 객체를 생성하는 작업을 각 서브클래스가 처리하도록 위임할 수 있습니다.

### 3. **추상 팩토리 패턴 (Abstract Factory Pattern)**

### **개념**

추상 팩토리 패턴은 관련된 객체들을 일관성 있게 생성할 수 있는 인터페이스를 제공합니다. 여러 개의 객체 생성에 대한 일관성을 보장하고, 특정 제품군을 생성하는 팩토리들을 추상화하는 패턴입니다.

### **언제 사용하나?**

- 서로 관련 있는 여러 객체를 생성해야 할 때 (제품군 생성)
- 플랫폼에 따라 서로 다른 객체를 생성해야 할 때 (예: Windows vs. MacOS GUI)

### **예시**

- **GUI 라이브러리**: 서로 다른 운영체제(Windows, MacOS, Linux)에서 작동하는 GUI 위젯을 일관되게 생성할 때, 추상 팩토리 패턴을 사용하여 각 운영체제에 맞는 위젯 팩토리를 제공할 수 있습니다.

### 4. **빌더 패턴 (Builder Pattern)**

### **개념**

빌더 패턴은 복잡한 객체를 단계별로 생성할 수 있도록 도와주는 패턴입니다. 객체의 생성 과정을 분리하고, 동일한 생성 절차에서 다양한 표현을 만들 수 있습니다.

### **언제 사용하나?**

- 복잡한 객체를 생성할 때, 특히 객체의 구성 요소가 많고 그 순서를 정확하게 제어해야 할 때
- 생성 과정에서 객체의 일부만 설정하거나 다양한 조합으로 객체를 생성해야 할 때

### **예시**

- **복잡한 보고서 생성**: 다양한 구성 요소(제목, 내용, 이미지, 표 등)를 갖는 보고서를 생성할 때 빌더 패턴을 사용하면 구성 요소의 생성과 조립 과정을 유연하게 관리할 수 있습니다.

### 5. **프로토타입 패턴 (Prototype Pattern)**

### **개념**

프로토타입 패턴은 기존 객체를 복제하여 새로운 객체를 생성하는 패턴입니다. 복잡한 객체를 생성하는 비용을 줄이고자 할 때 사용합니다.

### **언제 사용하나?**

- 객체 생성 비용이 높거나 복잡한 초기화가 필요한 경우
- 동일한 구조의 객체를 여러 번 생성해야 하는 경우

### **예시**

- **게임 캐릭터 생성**: 다양한 능력과 특성을 가진 게임 캐릭터를 만들 때 기본 캐릭터를 생성한 후, 이를 복제하여 다양한 특성을 추가하는 방식으로 캐릭터를 만들 수 있습니다.

### 생성 디자인 패턴 요약

| 패턴 | 설명 | 사용 상황 | 예시 |
| --- | --- | --- | --- |
| **싱글톤** | 하나의 인스턴스만 생성, 전역적으로 접근 가능 | 전역적으로 하나의 객체만 필요할 때 | 로깅, 설정 관리, DB 연결 |
| **팩토리 메서드** | 객체 생성을 서브클래스에 위임 | 객체 생성 로직이 자주 변경되거나, 클라이언트가 어떤 객체를 생성할지 모를 때 | 파일 형식별 문서 생성 |
| **추상 팩토리** | 관련된 객체들의 제품군을 생성하는 인터페이스 제공 | 관련된 객체들을 일관되게 생성할 때 | 운영체제별 GUI 라이브러리 |
| **빌더** | 복잡한 객체를 단계별로 생성 | 생성 과정이 복잡하고, 다양한 조합의 객체 생성이 필요할 때 | 복잡한 보고서, 차량 조립 |
| **프로토타입** | 기존 객체를 복제하여 새로운 객체 생성 | 객체 생성 비용이 높거나, 동일한 객체를 반복 생성할 때 | 게임 캐릭터, 문서 템플릿 |

### 결론

- **싱글톤 패턴**은 하나의 인스턴스를 공유해야 할 때 유용하며, 애플리케이션 전반에 걸쳐 자원을 효율적으로 관리할 수 있습니다.
- **팩토리 메서드 패턴**은 객체 생성 로직을 캡슐화하여 유연성을 높일 때 사용됩니다.
- **추상 팩토리 패턴**은 서로 관련된 객체들을 일관성 있게 생성해야 하는 상황에서 효과적입니다.
- **빌더 패턴**은 복잡한 객체를 단계별로 생성하고자 할 때 사용되어 객체 생성의 유연성과 가독성을 높입니다.
- **프로토타입 패턴**은 객체를 복제하여 생성하는 방식으로, 객체 생성의 비용과 시간을 줄이고자 할 때 유용합니다.

이러한 패턴들은 상황에 따라 적절하게 사용되어야 하며, 문제의 복잡도와 요구사항에 따라 어떤 패턴이 적합한지 판단하는 것이 중요합니다.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 샘플 시계열 데이터 생성
def generate_sample_data(length=1000, noise=0.2):
    t = np.linspace(0, 4 * np.pi, length)
    data = np.sin(t) + noise * np.random.randn(length)
    return data

# 시계열 데이터를 정규화하는 함수
def normalize_data(data):
    scaler = StandardScaler()
    normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return normalized

# 슬라이딩 윈도우로 데이터 분할
def sliding_window(data, window_size, step_size):
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)

# 시계열 데이터를 이미지로 변환
def time_series_to_image(data, window_size, step_size, save_dir='images', show_lines=True):
    windows = sliding_window(data, window_size, step_size)

    # 이미지 생성 및 저장
    for idx, window in enumerate(windows):
        plt.figure(figsize=(6, 4))
        plt.plot(window, label="Time Series", color='blue')
        plt.title(f"Window {idx}")
        plt.xlabel("Time")
        plt.ylabel("Value")

        # 보조선 추가
        if show_lines:
            for i in range(0, len(window), max(1, len(window) // 5)):
                plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
            plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        # 그래프 저장
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/window_{idx}.png")
        plt.close()

# 메인 실행
if __name__ == "__main__":
    import os

    # 샘플 데이터 생성
    data = generate_sample_data()
    normalized_data = normalize_data(data)

    # 파라미터 설정
    window_size = 100
    step_size = 50
    save_dir = 'images'

    # 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 시계열 데이터를 이미지로 변환
    time_series_to_image(normalized_data, window_size, step_size, save_dir, show_lines=True)

    print(f"Images saved to {save_dir}")