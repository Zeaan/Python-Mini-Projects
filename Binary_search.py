# Binary Search in Python3


def binary_search(array, x, low, high):

    while low <= high:

        mid = low + (high - low) // 2

        if array[mid] == x:
            return mid
        elif array[mid] < x:
            low = mid + 1
        else:
            high = mid - 1

    return -1


if binary_search([1, 2, 3, 4, 5, 6, 7], 3, 0, 6) != -1:
    print("Present")
else:
    print("Not Present")
