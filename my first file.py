from typing import Union


def summation(x, y: Union[int, float]) -> Union[int, float]:
    """
    сложение двух чисел
    :param x: int /  float
    :param y: int /  float
    :return:
        сумма двух чисел
    """

    try:
        int(x + y)
        print(x + y)
    except (TypeError, ValueError):
        print("Ошибка! Введите данные в числовом формате (int/float)")
    return x + y


summation(['a', 'b'], ['c', 'd'])
summation('a', 'd')
summation(1.7, 2.3)
summation(3, 4)
