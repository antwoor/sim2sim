from setuptools import setup, find_packages

setup(
    name="corrector",  # Название вашего пакета
    version="0.1",
    packages=find_packages(),  # Автоматически находит все подпакеты
    install_requires=[
        # Зависимости, например:
        "numpy",
        "gym",
        "torch",
    ],
    python_requires=">=3.10",  # Минимальная версия Python
    # Метаданные
    author="Geroyev Alexander",
    author_email="geroev_sasha@mail.ru",
    description="action corrector network for sim2sim and sim2real",
    license="GPL3",
    keywords="reinforcement-learning ppo mujoco, pybullet",
    
)