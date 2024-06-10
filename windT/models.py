from django.db import models

class Country(models.Model):
    name = models.CharField(max_length=100)
    country_description = models.TextField(default='There is no information yet')
    flag_photo = models.ImageField(upload_to='flags/')
    land_area = models.FloatField()

    def __str__(self):
        return self.name

class Year(models.Model):
    year = models.IntegerField()

    def __str__(self):
        return str(self.year)

class TotalCumulativeInstalledCapacity(models.Model):
    country = models.ForeignKey(Country, on_delete=models.CASCADE)
    year = models.ForeignKey(Year, on_delete=models.CASCADE)
    value = models.FloatField()

    def __str__(self):
        return f"{self.country.name} - {self.year.year} - {self.value} MW"

class GrowthRate(models.Model):
    country = models.ForeignKey(Country, on_delete=models.CASCADE)
    year = models.ForeignKey(Year, on_delete=models.CASCADE)
    value = models.FloatField()

    def __str__(self):
        return f"{self.country.name} - {self.year.year} - {self.value} %"