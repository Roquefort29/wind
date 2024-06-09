from django.db import models

class Country(models.Model):
    name = models.CharField(max_length=100)
    flag_photo = models.ImageField(upload_to='flags/')
    country_description = models.TextField(default='No description available')
    land_area = models.DecimalField(max_digits=15, decimal_places=2)

    def __str__(self):
        return self.name

class Year(models.Model):
    year = models.IntegerField(unique=True)

    def __str__(self):
        return str(self.year)

class TotalCumulativeInstalledCapacity(models.Model):
    country = models.ForeignKey(Country, on_delete=models.CASCADE)
    year = models.ForeignKey(Year, on_delete=models.CASCADE)
    value = models.DecimalField(max_digits=15, decimal_places=2)

    def __str__(self):
        return f"{self.country.name} - {self.year.year} - {self.value}"

class GrowthRate(models.Model):
    country = models.ForeignKey(Country, on_delete=models.CASCADE)
    year = models.ForeignKey(Year, on_delete=models.CASCADE)
    value = models.DecimalField(max_digits=6, decimal_places=2)

    def __str__(self):
        return f"{self.country.name} - {self.year.year} - {self.value}%"
