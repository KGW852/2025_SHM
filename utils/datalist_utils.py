# utils/datalist_utils.py


def remove_duplicates(dict_list, key_name):
    """
    Remove duplicate items based on file_name, etc., preserving the original order, and return a unique list
    """
    seen = set()
    unique_list = []
    for item in dict_list:
        key_value = item[key_name]
        if key_value not in seen:
            seen.add(key_value)
            unique_list.append(item)
    return unique_list