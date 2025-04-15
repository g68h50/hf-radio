class GenTools:
  def __init__(self, matrix_size, walsh_size):
    self.matrix_size = matrix_size
    self.walsh_size = walsh_size

  def __str__(self):
    return f"{self.matrix_size}({self.walsh_size})"